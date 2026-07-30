#set page(paper: "a4", margin: 2.2cm, numbering: "1 / 1")
#set text(font: ("Libertinus Serif", "DejaVu Serif"), size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show heading: it => block(above: 1.4em, below: 0.8em, it)
#show raw: it => text(font: "DejaVu Sans Mono", size: 0.88em, it)
#set table(stroke: 0.4pt + luma(65%), inset: 5pt)

// `mask` as a math operator, so it reads upright rather than as four variables.
#let mask = math.op("mask")
#let valid = math.op("valid")

// A null slot, drawn the same way everywhere so the diagrams read at a glance.
#let N = text(fill: rgb("#b03a2e"), weight: "bold", [NULL])

#let node(body, fill: luma(96%)) = box(
  inset: (x: 7pt, y: 5pt), radius: 3pt, stroke: 0.5pt + luma(55%), fill: fill, body,
)

#let lead(body) = block(
  inset: (x: 10pt, y: 8pt), radius: 3pt, fill: luma(97%),
  stroke: (left: 2pt + rgb("#2c3e50")), width: 100%, body,
)

#align(center)[
  #text(size: 17pt, weight: "bold")[Strictness and validity push-down]
  #v(-0.4em)
  #text(size: 12pt)[are the same property for Vortex scalar functions]
  #v(0.6em)
  #text(size: 9.5pt, fill: luma(35%))[
    Why one `is_strict` question is enough to license it, \
    and why totality and dense-safety are two further, independent axes
  ]
]

#v(1em)

#lead[
  *Summary.* Vortex documents strictness as null propagation, while the optimizer actually relies on
  an identity about moving *validity* through a function. This note proves the two are equivalent for
  any row-local function, so a single `is_strict` answer licenses every validity push-down. It then isolates the two things that do *not* follow:
  the obligation that the return dtype can represent a null, and *totality*, which is what makes
  output validity precomputable.
]

= Setup <setup>

Every scalar function in Vortex is *row-local*: row $i$ of the output is computed from row $i$ of the
inputs and nothing else. Writing $f(a_1, ..., a_k)[i]$ for the output at row $i$,

$ f(a_1, ..., a_k)[i] "depends only on" (a_1 [i], ..., a_k [i]). $

This is not an assumption about well-behaved functions, it is what `ScalarFnVTable::execute` computes.
Aggregates are not row-local, which is why none of what follows applies to them.

The property below is about *validity*: which rows of a column are null. A *mask* is just how Vortex
represents and applies validity, a non-nullable boolean array that nulls out the rows where it is false
and leaves the rest alone:

$ mask(a, m)[i] = cases(#N &"if" not m[i], a[i] &"otherwise") $

Note that a masked-out slot still *holds* whatever byte pattern it held before. Only its validity
changed. That distinction is what @dense returns to.

#figure(
  table(
    columns: 4,
    align: center,
    table.header([$i$], [$a$], [$m$], [$mask(a, m)$]),
    [0], [10],       [`true`],  [10],
    [1], [20],       [`false`], N,
    [2], N,          [`true`],  N,
    [3], [40],       [`false`], N,
  ),
  caption: [Applying validity. Row 1 was valid and is nulled out. Row 2 was already null and stays null. After
    masking there is no way to tell those two rows apart, which is the hinge of @equiv.],
)

= The two properties

We are interested in two statements about $f$, which the codebase has historically conflated.

#lead[
  *(S) Strict, or null-propagating.* For every row $i$: if $a_j [i] = #N$ for some $j$, then
  $f(a_1, ..., a_k)[i] = #N$.
]

This is the PostgreSQL `STRICT` reading, and it is how `ScalarFnVTable::is_strict` is documented.
It says nothing about rows where every input is non-null.

#lead[
  *(M) Per-argument validity equivariance.* For every argument position $j$, every input tuple, and every
  mask $m$:
  $ f(a_1, ..., mask(a_j, m), ..., a_k) = mask(f(a_1, ..., a_j, ..., a_k), m) $
]

Read it as: *applying validity to one input, then computing, is the same as computing, then applying
that validity to the output.* This is the identity every validity push-down relies on (@pushdown), and
@naming says what to call it and what actually moves.

Here is (M) holding for $f = "add"$, applying validity to the left argument only:

#figure(
  table(
    columns: 6,
    align: center,
    table.header(
      [$i$], [$a_1$], [$a_2$], [$m$],
      [mask first, \ then add], [add first, \ then mask],
    ),
    [0], [1], [10], [`true`],  [11], [11],
    [1], [2], [20], [`false`], N,    N,
    [2], [3], [30], [`false`], N,    N,
  ),
  caption: [The two sides agree row by row. On row 1 the left side computes $"add"(#N, 20) = #N$ by
    strictness, and the right side computes $"add"(2, 20) = 22$ and then masks it away.],
)

= (S) and (M) are equivalent <equiv>

#lead[
  *Theorem.* For a row-local $f$ over inhabited dtypes, (S) holds if and only if (M) holds.
]

== (S) implies (M)

Fix an argument position $j$, an input tuple, a mask $m$, and a row $i$. Because both sides are
row-local, it suffices to check row $i$, and there are only two cases.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (center, left, left),
    table.header([case], [left-hand side at row $i$], [right-hand side at row $i$]),
    [$m[i] = $ `true`],
    [$mask(a_j, m)[i] = a_j [i]$, so this is \ $f(a_1 [i], ..., a_j [i], ..., a_k [i])$],
    [masking with a true bit is the identity, \ so this is the same value],

    [$m[i] = $ `false`],
    [$mask(a_j, m)[i] = #N$, \ so (S) forces #N],
    [$mask(dot, m)[i] = #N$ \ by definition],
  ),
  caption: [Both cases agree, so the columns are equal. #h(0.5em) $square.stroked$],
)

== (M) implies (S)

By contraposition: assume $f$ is *not* strict and derive a violation of (M).

Not strict means there is an input tuple $b_1, ..., b_k$ and a row $i$ with $b_j [i] = #N$ for some
$j$, yet $f(b_1, ..., b_k)[i] = w$ for some $w != #N$.

The trick is to *manufacture* $b_j$ by masking, so that (M) has something to say about it. Let $a'_j$
agree with $b_j$ everywhere except row $i$, where it holds an arbitrary non-null value $u$. Let $m$
be false at row $i$ and true everywhere else. Then $mask(a'_j, m)$ is $b_j$ exactly:

#figure(
  table(
    columns: 5,
    align: center,
    table.header([row], [$b_j$ (given)], [$a'_j$ (built)], [$m$ (built)], [$mask(a'_j, m)$]),
    [$dots.v$], [$x$], [$x$], [`true`],  [$x$],
    [$i$],      N,     [$u$], [`false`], N,
    [$dots.v$], [$y$], [$y$], [`true`],  [$y$],
  ),
  caption: [Off row $i$ the mask is true and passes $a'_j = b_j$ through. At row $i$ both are null.
    So $mask(a'_j, m) = b_j$, and any statement (M) makes about $a'_j$ constrains $b_j$.],
)

Now instantiate (M) at position $j$, inputs $b_1, ..., a'_j, ..., b_k$, mask $m$, and read off row $i$:

$
"LHS"[i] &= f(b_1, ..., mask(a'_j, m), ..., b_k)[i] = f(b_1, ..., b_j, ..., b_k)[i] = w != #N \
"RHS"[i] &= mask(f(b_1, ..., a'_j, ..., b_k), m)[i] = #N #h(1em) "since" m[i] = "false"
$

So (M) would force $w = #N$, contradicting $w != #N$. Hence (M) implies (S). $square.stroked$

#lead[
  The reverse direction needs a non-null $u$ to exist, that is, the dtype must be inhabited. Every
  Vortex dtype is, so the equivalence holds unconditionally in practice.
]

== Why the proof needs row-locality

Both directions used it. If $f$ could read other rows, masking row $i$ could change the output at row
$i' != i$, and neither implication would go through. Strictness and validity equivariance are equivalent
*for scalar functions specifically*.

= A note on names, and on what actually moves <naming>

Three things get conflated here, so it is worth separating them.

*(M) is a law, not a rewrite.* It is an equality, so it has no direction: it is simply true or false of
a given $f$. An optimizer *rewrite* has a direction and picks whichever side is cheaper. So (M) is
stated separately from the push-downs it licenses in @pushdown.

*The subject is validity, not the mask.* A mask is one representation of validity, so a law phrased in
terms of $mask$ can read as though it were about boolean arrays. It is not. (M) says that $f$ respects
the *action* of applying validity, $f(m dot x) = m dot f(x)$, which is equivariance, and is the same
shape as a homomorphism condition. Vortex's older documentation said $f$ "distributes over `mask`",
which is the same idea in plainer words.

*What moves is the function, not the validity.* Reading `dict/compute/rules.rs`: before the rewrite a
null code supplies null for that argument while the sibling constants keep their values, so $f$ sees
the validity. After it, the null code applies to $f$'s result instead. The codes array is never
touched, so the validity does not physically move at all. What moves is $f$, pushed down past the
validity to sit directly on the dictionary values:

#align(center)[
  #grid(
    columns: 3, column-gutter: 1.2em, align: horizon,
    node[$f(mask(x, m), c)$ #h(0.4em) #text(size: 8pt, fill: luma(45%))[$f$ sees validity]],
    text(size: 13pt)[$arrow.r.long$],
    node(fill: rgb("#eafaf1"))[$mask(f(x, c), m)$ #h(0.4em) #text(size: 8pt, fill: luma(45%))[$f$ does not]],
  )
]

That is what makes *validity push-down* the right name for the family: computation is pushed down past
validity, to where the data actually lives. The codebase describes the same rewrite from the other end,
as pushing the scalar function into the dictionary values.

Note that (M) is read in both directions in Vortex. The dictionary rewrite uses it left to right, to
get $f$ off the decoded column. `NullHandling::Filter` (@dense) uses it right to left: filter the
inputs to the rows valid in every input, run the kernel over fewer rows, then scatter the results back.

= What does not follow: the dtype obligation

(S) and (M) are statements about *values*. Neither says the output can *represent* the null it
demands, and that is a real, separate obligation.

`cast` is the witness. It is value-strict, and hence validity-equivariant, but its options can pin a
non-nullable output dtype. Then the null that (S) requires has nowhere to go:

#figure(
  table(
    columns: 4,
    align: (center, center, center, left),
    table.header([$i$], [input `i32?`], [`cast(_, i64)` \ non-nullable output], [what happens]),
    [0], [1], [1], [fine],
    [1], N,   [#text(fill: rgb("#b03a2e"))[?]],
      [(S) demands #N, the dtype forbids it],
  ),
  caption: [Value-level strictness is not enough. `cast` must answer `is_strict = false`.],
)

So the property the vtable actually advertises is a conjunction:

$ "is_strict" eq.triple underbrace((S) equiv (M), "proved equivalent above") and underbrace("output dtype admits those nulls", "an obligation, not a consequence") $

`StrictScalarFnVTable` discharges the second conjunct *structurally*: its `return_dtype` is the
kernel's element dtype widened to nullable if and only if some input dtype is nullable. An
implementor cannot get it wrong, which is why the blanket impl may answer `is_strict = true`
unconditionally.

= Per-argument, not all-arguments

(M) masks *one* argument. The weaker law masks them all at once, and the difference is not academic.

Kleene `AND` satisfies the all-arguments law but violates (M). Mask only the second argument:

#figure(
  table(
    columns: 6,
    align: center,
    table.header(
      [$i$], [$a_1$], [$a_2$], [$m$],
      [mask $a_2$, \ then `AND`], [`AND`, \ then mask],
    ),
    [0], [`false`], [`true`], [`false`],
      table.cell(fill: rgb("#fdecea"))[`false`],
      table.cell(fill: rgb("#eafaf1"))[#N],
  ),
  caption: [$"false" and N = "false"$ under Kleene logic, so the left side keeps a non-null value
    where the right side is null. The two disagree, so Kleene `AND` is not strict.],
)

Push-down masks one child while leaving sibling constants untouched, so it is (M), the per-argument
form, that has to hold. This is why the documented law is stated per argument.

= Totality: a third, independent property

Neither (S) nor (M) constrains $f$ at rows where *every* input is non-null. A function may return
null there and remain perfectly strict.

#lead[
  *(T) Total.* For every row $i$: if $a_j [i] != #N$ for all $j$, then $f(a_1, ..., a_k)[i] != #N$.
]

`list_sum` is strict but not total, because summing a *valid, empty* list is null:

#figure(
  table(
    columns: 4,
    align: center,
    table.header([$i$], [`list` input], [`list_sum`], [why]),
    [0], [`[1, 2]`], [3],  [ordinary row],
    [1], [`[]`],     N,    [valid input, null output: not total],
    [2], N,          N,    [null input, null output: strict],
  ),
  caption: [Rows 1 and 2 are both null in the output, for entirely different reasons.],
)

Such a null cannot break (M), because it appears on *both* sides of the equation and cancels. Take
row 1 above with $m[1] = $ `true`:

$ "LHS"[1] = #N #h(2em) "RHS"[1] = mask(#N, m)[1] = #N $

So (S) gives an *inclusion* on validity rather than an equality, and (T) is exactly the missing
reverse direction:

#figure(
  grid(
    columns: 3, column-gutter: 1.1em, align: horizon,
    node(fill: rgb("#eaf2f8"))[$valid(f(a_1, ..., a_k))$],
    [$subset.eq$],
    node(fill: rgb("#eaf2f8"))[$valid(a_1) and ... and valid(a_k)$],
  ),
  caption: [What strictness alone buys. Equality holds if and only if $f$ is also total.],
)

The practical consequence is narrow and worth stating precisely:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  table.header([given], [what you may conclude]),
  [strict], [the inclusion above, so every validity push-down is sound],
  [strict *and* total], [the equality, so output validity can be *precomputed* without executing $f$],
)

A non-total strict function therefore loses exactly one thing: the validity shortcut. It keeps every
push-down. In Vortex this is expressed by leaving `StrictScalarFnVTable::validity` at its `None`
default, so the unsound answer is the one that takes work to write.

= Consequences <pushdown>

== Dictionary push-down needs (M), which is why `is_strict` suffices

Dictionary push-down rewrites a function over dictionary-encoded input so that it evaluates over the
values array, which is usually far smaller than the decoded column:

#align(center)[
  #grid(
    columns: 3, column-gutter: 1.2em, align: horizon,
    node[`f(dict(codes, values), c)`],
    text(size: 13pt)[$arrow.r.long$],
    node(fill: rgb("#eafaf1"))[`dict(codes, f(values, c))`],
  )
]

On the right, `f` never sees the codes' nulls. They are reapplied by the surrounding `dict`, which is
exactly $mask(f(...), m)$ set against the original $f(mask(...), c)$. Worked through, with
$f = "add"(dot, 10)$ over `values = [2, 5]`, where the codes array is nullable and `values` is not:

#figure(
  table(
    columns: 5,
    align: center,
    table.header(
      [$i$], [`codes`], [decoded column],
      table.cell(fill: rgb("#fdf2e9"))[original \ `add(decoded, 10)`],
      table.cell(fill: rgb("#eafaf1"))[pushed down \ `dict(codes, add(values, 10))`],
    ),
    [0], [0], [2],  [12], [12],
    [1], N,   N,    N,    N,
    [2], [1], [5],  [15], [15],
  ),
  caption: [The two result columns agree. Row 1 is null on the left because `add` propagated the
    decoded null, and null on the right because the outer `dict` reapplied the code's null. Those are
    the two sides of (M). Note that `add(values, 10)` is computed over just two rows, not three.],
)

The rewrite is sound exactly when (M) holds for the dictionary argument alone, with the sibling
constant `c` left unmasked. By @equiv, asking the single value-level question `is_strict` answers it,
so no separate equivariance flag is needed on the vtable.

== Errors are a fourth axis that (M) says nothing about

The rewritten tree evaluates `f` over *every* dictionary value, including entries that no live code
references. If `f` can fail on some of those values, the rewrite manufactures an error the original
query would never have raised. (M) is a statement about values, not about failure, so it cannot rule
this out.

Take $f = "div"(100, dot)$ over `values = [4, 0]`, where every live code happens to reference index 0.
Push-down evaluates `f` per *value*, not per row, so the comparison is over the values array:

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    table.header(
      [`values` index], [value], [reached by a live code?],
      table.cell(fill: rgb("#fdf2e9"))[original evaluates], table.cell(fill: rgb("#eafaf1"))[pushed down evaluates],
    ),
    [0], [`4`], [yes], [`div(100, 4)` #sym.arrow.r `25`], [`div(100, 4)` #sym.arrow.r `25`],
    [1], [`0`], [no], [nothing],
      table.cell(fill: rgb("#fdecea"))[`div(100, 0)` #sym.arrow.r *error*],
  ),
  caption: [Value 1 is dead, so the original query never divides by zero, while the pushed-down form
    evaluates every value and fails. This is why push-down gates on `is_strict && !is_fallible`, and
    why an element that *parses* its bytes (a WKB geometry) has to report fallibility even when its
    own kernel cannot fail.],
)

== `RowFn` guarantees (S) structurally <dense>

A function built on `RowFn` cannot be non-strict, on either execution path:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  table.header([path], [why (S) holds by construction]),
  [`NullHandling::Filter`],
  [the kernel is handed only rows valid in every input, and the lifting writes #N at every
   other row],
  [`NullHandling::Dense`],
  [the kernel runs over everything, then the lifting masks the result with
   $valid(a_1) and ... and valid(a_k)$, so any row null in any input becomes #N regardless of
   what the kernel returned],
)

Together with the structural `return_dtype`, both conjuncts of `is_strict` are discharged by the
framework, so `RowFn` needs no `is_strict` member at all.

The `Dense` path is where the *held-but-invalid* byte pattern from @setup matters. The kernel
does read the slot behind a null row, so that read has to be safe. Whether it is depends on the
element, not the function: a value sitting in a flat buffer is garbage but harmless, whereas a string
view behind a null row may name a buffer that does not exist. This is tracked per element as
`InputElement::DENSE_SAFE`, and it is a purely operational concern, entirely separate from the
semantic question the proof above settles.

= The axes, separated

#table(
  columns: (auto, 1fr, 1fr),
  align: (left, left, left),
  table.header([property], [a statement about], [what it unlocks]),
  [*strict* $equiv$ (M)], [values at rows with a null input],
    [every validity push-down, including the dictionary rewrite],
  [*total*], [values at rows with no null input],
    [precomputing output validity ($subset.eq$ becomes $=$)],
  [*fallible*], [whether legal values can raise an error],
    [speculative evaluation, as in dictionary push-down],
  [*dense-safe*], [whether the bytes behind a null row can be read without faulting],
    [skipping the filter-and-scatter round trip],
)

The first is proved equivalent to the law the optimizer wants. The other three are genuinely
independent of it, and of each other, which is why each is tracked separately rather than inferred.
