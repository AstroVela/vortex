# The same functions in both systems

Each section writes one function in both frameworks and points at what the difference in shape
reveals. Vortex code comes from the `ct/row-fn-*` stack. Velox code comes from
`facebookincubator/velox` at `54fea71cc`, lightly trimmed.

## 1. The minimal function: `hypot`

Vortex, from #9129 (the epic's running example):

```rust
impl RowFn for Hypot {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["x", "y"];
    const FALLIBLE: bool = false;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.hypot");
        *ID
    }

    fn dispatch<V: RowVisitor<Self::Options>>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::VisitResult> {
        visitor.visit::<(f64, f64), f64>(|(x, y)| x.hypot(y))
    }
}
```

Velox, in the shape of `velox/functions/prestosql/Arithmetic.h`:

```cpp
template <typename T>
struct HypotFunction {
  FOLLY_ALWAYS_INLINE void call(double& result, const double& x, const double& y) {
    result = std::hypot(x, y);
  }
};

// At startup, once per signature:
registerFunction<HypotFunction, double, double, double>({"hypot"});
```

What the shapes reveal:

- The Velox author writes no types beyond the C++ signature. Arity, argument types, and return type
  all come from the `call` signature plus the registration call. The Vortex author writes the arity
  (`ARG_NAMES`), the fallibility (`FALLIBLE`), an identity (`id`), and the type selection
  (`visit::<(f64, f64), f64>`), because a single registered `RowFn` must answer for every input
  dtype at runtime.
- The Velox function has a name only at registration. The Vortex function owns a `ScalarFnId`
  because Vortex expressions serialize into files and must resolve back to the function.
- Velox needs one `registerFunction` call per accepted signature. The Vortex `dispatch` accepts or
  rejects dtypes with ordinary code, so "which signatures exist" is a runtime property.

## 2. One definition, many widths

Both systems monomorphize the row closure per element width. They drive the monomorphization from
opposite ends.

Velox stamps the template at registration
(`velox/functions/lib/RegistrationHelpers.h`):

```cpp
template <template <class> typename T>
void registerBinaryIntegral(const std::vector<std::string>& aliases) {
  registerFunction<T, int8_t, int8_t, int8_t>(aliases);
  registerFunction<T, int16_t, int16_t, int16_t>(aliases);
  registerFunction<T, int32_t, int32_t, int32_t>(aliases);
  registerFunction<T, int64_t, int64_t, int64_t>(aliases);
}
```

Vortex stamps generic functions inside `dispatch`
(`vortex-array/src/scalar_fn/fns/binary/compare/primitive.rs`):

```rust
fn dispatch<V: RowVisitor<Self::Options>>(
    &self,
    op: &Self::Options,
    args: &[DType],
    visitor: V,
) -> VortexResult<V::VisitResult> {
    let ptype = PType::try_from(args.first().ok_or_else(|| {
        vortex_err!("a comparison operator takes two operands, got none")
    })?)?;

    match_each_native_ptype!(ptype, |T| { visit_compare::<T, V>(*op, visitor) })
}

fn visit_compare<T, V>(op: CompareOperator, visitor: V) -> VortexResult<V::VisitResult>
where
    T: NativePType,
    V: RowVisitor<CompareOperator>,
{
    match op {
        CompareOperator::Eq => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_eq(rhs)),
        CompareOperator::Gt => visitor.visit::<(T, T), bool>(|(lhs, rhs)| lhs.is_gt(rhs)),
        // ... four more operators
    }
}
```

The generated code is close to equivalent: one specialized loop per `(width, operator)` pair. The
difference is where the runtime branch lives. Velox branches in the registry, once per compiled
expression. Vortex branches in `match_each_native_ptype!` and the operator match, once per batch.
The Vortex form also runs `dispatch` at least twice per call (once to plan the output dtype, once
per execution attempt), and the executor verifies the runs agree (`ensure_plan`).

## 3. Checked arithmetic: where the error channel lives

This is the sharpest behavioral difference between the two frameworks.

Velox Presto `plus` uses a checked helper that throws
(`velox/functions/prestosql/Arithmetic.h`):

```cpp
template <typename T>
struct PlusFunction {
  template <typename TInput>
  FOLLY_ALWAYS_INLINE void call(TInput& result, const TInput& a, const TInput& b) {
    result = plus(a, b);  // checkedPlus<int64_t> throws VeloxUserError on overflow
  }
};
```

The adapter wraps every row loop in `applyToSelectedNoThrow`, which catches the throw and records a
per-row error in `EvalCtx`. A surrounding `TRY` turns that row's error into a null. Rows after the
failing row still execute. The cost model: the happy path pays for a try/catch region and a
per-row branch inside `checkedPlus`, and a throwing row pays the full C++ exception machinery.

Vortex checked add cannot afford a per-row branch in the loop, because the branch blocks
vectorization. It returns evidence instead
(`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs`):

```rust
visitor.visit_deferred::<(T, T), T, Op::Fail>(
    |(lhs, rhs)| Op::apply(lhs, rhs),          // returns (value, evidence), branch-free
    |failure| {
        if failure != <Op::Fail as Default>::default() {
            return Err(numeric_error(Op::ERROR)); // built once, after the loop
        }
        Ok(())
    },
)
```

The executor OR-reduces the evidence in a loop-local accumulator and constructs one error after the
loop. Because the `Dense` policy also evaluated the garbage behind null rows, that error can be a
false positive. The batch layer then retries only the valid rows (`DenseWithRetry`) before it
reports anything. A `const` assertion pins `size_of::<Fail>() <= size_of::<Out>()` so the evidence
never bounds the vector width, and the evidence type is width-dependent: `bool` where a flag is
free, a full word where narrowing costs the vectorization it guards. For 64-bit multiply the
evidence is the XOR of the discarded high half against the sign extension of the kept half.

There is no Velox analogue to any of this, and there is no Vortex analogue to `TRY`: a Vortex batch
with one failing valid row fails as a batch. Velox reports the failing row.

## 4. Integer division: immediate errors and uninitialized output

Both systems agree that division is different: it is scalar and expensive anyway, so a per-row check
is free.

Velox can return `Status` instead of throwing (`velox/functions/prestosql/Fail.h` shows the
convention):

```cpp
FOLLY_ALWAYS_INLINE Status
call(out_type<UnknownValue>& /*result*/, const arg_type<Varchar>& message) {
  return Status::UserError("{}", message);
}
```

Vortex stops at the first failure and writes into uninitialized storage, with a token proving each
write happened (`fns/binary/numeric/row.rs`):

```rust
visitor.visit_into::<(T, T), UninitElementSink<T>, _>(|(lhs, rhs), output| {
    let (value, failed) = CheckedDiv::apply(lhs, rhs);
    if failed {
        return Err(numeric_error(<CheckedDiv as CheckedPrimitiveOp<T>>::ERROR));
    }

    // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
    Ok(unsafe { InitializedElement::write(output, value) })
})
```

The Velox output slot always exists (the adapter called `ensureWritable` and cleared nulls up
front). The Vortex `UninitElementSink` skips initializing the output at all, and the type system
carries the proof that a successful callback wrote its slot: `InitializedElement` has no safe
constructor, which a `compile_fail` doctest pins.

## 5. Hoisting constant-argument work

Both frameworks solve the same problem: work derived from a constant argument must run once, not
once per row. The mechanisms differ in scope and in state.

Velox `date_trunc` pre-parses a constant unit string once per query and thread
(`velox/functions/prestosql/DateTimeFunctions.h`):

```cpp
const tz::TimeZone* timeZone_ = nullptr;
std::optional<DateTimeUnit> unit_;

FOLLY_ALWAYS_INLINE void initialize(
    const std::vector<TypePtr>& /*inputTypes*/,
    const core::QueryConfig& config,
    const arg_type<Varchar>* unitString,   // non-null iff the argument is a plan constant
    const arg_type<Timestamp>* /*timestamp*/) {
  timeZone_ = getTimeZoneFromConfig(config);
  if (unitString != nullptr) {
    unit_ = getTimestampUnit(*unitString);
  }
}

FOLLY_ALWAYS_INLINE void call(out_type<Timestamp>& result,
    const arg_type<Varchar>& unitString, const arg_type<Timestamp>& timestamp) {
  DateTimeUnit unit = unit_.has_value() ? unit_.value()
                                        : getTimestampUnit(unitString).value();
  result = truncateTimestamp(timestamp, unit, timeZone_);
}
```

Vortex `CosineSimilarity` hoists the norm of a constant operand once per batch
(`vortex-tensor/src/scalar_fns/cosine_similarity.rs`):

```rust
visitor.visit_prepared_into::<(TensorRow<T>, TensorRow<T>), UninitElementSink<T>, _, _>(
    |(lhs, rhs)| ConstNorms {
        lhs: lhs.map(l2_norm_row),   // Some(elem) iff the operand is batch-constant
        rhs: rhs.map(l2_norm_row),
    },
    |norms, (lhs, rhs), output| {
        // SAFETY: `output` is the `UninitElementSink` row supplied for this callback.
        unsafe {
            InitializedElement::write(output, cosine_similarity_row_prepared(norms, lhs, rhs))
        }
    },
)
```

Differences that matter:

- **Scope.** Velox `initialize` runs once per query and thread, and sees only plan-time constant
  literals. Vortex `prepare` runs once per batch, and sees any operand whose array is a
  `ConstantArray` at execution time, including constants that an encoding produced mid-plan. Velox
  does still handle runtime-constant vectors cheaply (a constant reader indexes every row at 0),
  but derived work such as parsing or norm computation only hoists for plan constants.
- **State.** The Velox function stores hoisted state in mutable member fields, and the framework
  creates one function instance per compiled expression to make that safe. The Vortex `prepare`
  closure returns a typed value that the executor passes to the row closure by reference. The row
  closure is `Fn`, not `FnMut`, so it cannot mutate shared state at all. That constraint is
  load-bearing: a captured `&mut` measured 8 to 11% slower because the mutable capture blocked
  loop vectorization.
- **Laziness.** Vortex `SpatialContains` goes one step further and prepares a constant geometry's
  topology graph lazily in a `OnceCell`, on the first row that needs it, because eager preparation
  charges point-only batches for nothing. Velox `initialize` has no lazy form. A Velox function has
  to hand-roll the same `folly::once`-style pattern in a member.
- **Failure.** A throwing Velox `initialize` is captured and replayed as a per-row error only if
  active rows exist, to keep "all inputs null" returning null instead of an error. Vortex `prepare`
  is infallible by design. Anything fallible belongs in `decode` or in the row result.

## 6. Nullable outputs: the open question, answered two ways

Presto `array_min` returns null for an empty array and for an array containing null. In Velox that
is one `bool` return (`velox/functions/prestosql/ArrayFunctions.h`):

```cpp
bool call(out_type<Orderable<T1>>& out, const arg_type<Array<Orderable<T1>>>& array) {
  if (array.size() == 0) {
    return false;              // null result from a valid input
  }
  // ... returns false on null element, true otherwise
}
```

`RowFn` cannot express this function today. Its contract is stronger than strictness: a row kernel
must produce a valid value for every valid input row, and the framework derives output validity
entirely from input validity. #9128 excludes `list_sum` and `variant_get` for exactly this reason
and leaves nullable row outputs as the main unresolved question.

The Velox design shows both the benefit and the bill. Benefit: row-level null decisions with no
framework change, and `can_produce_null_output` derived from the return type. Bill: every fast-path
row write goes through a `notNull` branch, functions that can produce null are excluded from
result-vector reuse and from the engine-wide flat-no-nulls fast path, and the adapter must
pessimistically clear result nulls up front. Vortex's stricter contract is what lets its dense loop
be a straight-line `map` into uninitialized memory.

## 7. Beyond the current RowFn surface

Three Velox capabilities have no `RowFn` counterpart yet. Each is a preview of what an
`InputElement`/`OutputSink` extension needs to cover.

**Strings, with ASCII specialization and zero-copy output** (`StringFunctions.h`, `substr`):

```cpp
template <typename T>
struct SubstrFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);
  static constexpr int32_t reuse_strings_from_arg = 0;   // result aliases arg 0's buffers
  static constexpr bool is_default_ascii_behavior = true; // ASCII in implies ASCII out

  template <typename I>
  FOLLY_ALWAYS_INLINE void call(out_type<Varchar>& result,
      const arg_type<Varchar>& input, I start, I length) {
    doCall<false>(result, input, start, length);
  }

  template <typename I>
  FOLLY_ALWAYS_INLINE void callAscii(out_type<Varchar>& result,
      const arg_type<Varchar>& input, I start, I length) {
    doCall<true>(result, input, start, length);         // byte == character
  }
  // doCall ends with: result.setNoCopy(StringView(input.data() + range.first, ...));
};
```

The engine scans string inputs once per batch, dispatches to `callAscii` when all are ASCII, and
attaches the input's string buffers to the result vector so `setNoCopy` views stay alive. The
`RowFn` analogues are: a per-batch input property in the plan, a second dispatch arm, and an
`OutputSink` that holds a reference on an input buffer.

**Complex outputs through writers** (`ArrayFunctions.h`, `array_cum_sum`):

```cpp
FOLLY_ALWAYS_INLINE void call(out_type<velox::Array<T>>& out,
    const arg_type<velox::Array<T>>& in) {
  NativeType sum = 0;
  for (auto i = 0; i < in.size(); ++i) {
    if (in[i].has_value()) {
      sum = checkedPlus<NativeType>(sum, in[i].value());
      out.add_item() = sum;
    } else {
      for (auto j = i; j < in.size(); ++j) {
        out.add_null();
      }
      break;
    }
  }
}
```

`out` writes directly into the child vectors of the result `ArrayVector`. This is the general form
of `OutputSink`: variable-width output built in place. Note `add_null()`: complex writers can emit
nulls inside a row's value, which is orthogonal to the row itself being null.

**Variadic and generic signatures** (`concat(array...)` takes 2 to 252 arguments,
`cardinality` takes `Array<Generic<T1>>` or `Map<Generic<T1>, Generic<T2>>`). `RowFn` fixes arity
at `ARG_NAMES.len()` with sealed tuples through arity 12, and expresses genericity by dispatching
on dtypes instead. Velox's registry needs a priority lattice (concrete beats variadic beats generic
beats variadic-of-generic) to keep resolution deterministic once these features exist. Vortex's
dispatch-side binding gets the same effect with a `match`.

## 8. Escape hatches

Both frameworks accept that some kernels beat the row form, and both keep the row form as the
default.

Vortex puts the first hatch inside the function. `L2Norm` reads stored norms straight off a
`Normalized` encoding (`vortex-tensor/src/scalar_fns/l2_norm.rs`):

```rust
fn reduce_encoded(&self, _options: &Self::Options, args: &[ArrayRef], _ctx: &mut ExecutionCtx)
    -> VortexResult<Option<RowExecution>> {
    let input = &args[0];
    if input.is::<Normalized>() {
        let (_, norms) = extract_normalized_children(input);
        // ... dtype checks ...
        return Ok(Some(RowExecution::Output(norms)));
    }
    Ok(None)
}
```

The second Vortex hatch sits above the function: `compare_primitive_with_path` routes measured
cases to a fused columnar compare-and-bitpack kernel, per ptype, operator, constness, and
architecture, and keeps the row path for everything else.

Velox's hatch is a separate registration. `is_null` flips the nulls buffer in bulk
(`velox/functions/lib/IsNull.cpp`), subscript returns a dictionary view over the elements vector
without copying (`SubscriptUtil.h`), and for comparisons an xsimd `ComparisonSimdFunction` covers
the fixed-width types and outranks the simple functions at resolution
(`velox/functions/prestosql/Comparisons.cpp`):

```cpp
template <typename ComparisonOp, typename Arch = xsimd::default_arch>
class ComparisonSimdFunction : public exec::VectorFunction {
  void apply(const SelectivityVector& rows, std::vector<VectorPtr>& args,
      const TypePtr& outputType, exec::EvalCtx& context, VectorPtr& result) const override {
    // ... xsimd batch compare into the boolean result bits ...
  }
  bool supportsFlatNoNullsFastPath() const override { return true; }
};
```

The parallel is exact. Both projects found that a comparison producing one bit per row wants a
compare-and-movemask loop, that a one-value-per-row API cannot express it, and that the right fix
is a columnar implementation selected ahead of the row one. Velox selects it in the registry.
Vortex selects it in a hand-written dispatch function with a benchmark citation in a comment.
