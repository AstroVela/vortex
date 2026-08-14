// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Extension {
    fn cast(array: ArrayView<'_, Extension>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        if !array.dtype().eq_ignore_nullability(dtype) {
            let DType::Extension(target_ext_dtype) = dtype else {
                return Ok(Some(array.storage_array().cast(dtype.clone())?));
            };

            let source_ext_dtype = array.dtype().as_extension();

            // `can_coerce_from` may require an extension-specific value conversion. This generic
            // cast only supports `can_coerce_to`, where casting the storage is sufficient.
            if !source_ext_dtype.can_coerce_to(dtype) {
                return Ok(None);
            }

            let target_storage = array
                .storage_array()
                .cast(target_ext_dtype.storage_dtype().clone())?;

            return Ok(Some(
                ExtensionArray::new(target_ext_dtype.clone(), target_storage).into_array(),
            ));
        }

        let DType::Extension(ext_dtype) = dtype else {
            unreachable!("Already verified we have an extension dtype");
        };

        let new_storage = match array
            .storage_array()
            .cast(ext_dtype.storage_dtype().clone())
        {
            Ok(arr) => arr,
            Err(e) => {
                tracing::warn!("Failed to cast storage array: {e}");
                return Ok(None);
            }
        };

        Ok(Some(
            ExtensionArray::new(ext_dtype.clone(), new_storage).into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::vortex_ensure;
    use vortex_session::VortexSession;

    use super::*;
    use crate::EmptyMetadata;
    use crate::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::compute::conformance::cast::test_cast_conformance;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::extension::ExtDType;
    use crate::dtype::extension::ExtId;
    use crate::dtype::extension::ExtVTable;
    use crate::executor::VortexSessionExecute;
    use crate::extension::datetime::TimeUnit;
    use crate::extension::datetime::Timestamp;
    use crate::scalar::ScalarValue;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(crate::array_session);

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    struct MillisecondTimestamp;

    impl ExtVTable for MillisecondTimestamp {
        type Metadata = EmptyMetadata;
        type NativeValue<'a> = &'a ScalarValue;

        #[expect(clippy::disallowed_methods, reason = "test-only extension ID")]
        fn id(&self) -> ExtId {
            ExtId::new("vortex.test.millisecond_timestamp")
        }

        fn serialize_metadata(&self, _metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
            Ok(Vec::new())
        }

        fn deserialize_metadata(&self, _metadata: &[u8]) -> VortexResult<Self::Metadata> {
            Ok(EmptyMetadata)
        }

        fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
            vortex_ensure!(
                matches!(ext_dtype.storage_dtype(), DType::Primitive(PType::I64, _)),
                "MillisecondTimestamp storage must be i64, got {}",
                ext_dtype.storage_dtype(),
            );
            Ok(())
        }

        fn can_coerce_to(source: &ExtDType<Self>, target: &DType) -> bool {
            let Some(target) = target.as_extension_opt() else {
                return false;
            };
            let Some(options) = target.metadata_opt::<Timestamp>() else {
                return false;
            };

            options.unit == TimeUnit::Milliseconds
                && options.tz.is_none()
                && target
                    .storage_dtype()
                    .can_coerce_from(source.storage_dtype())
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    #[test]
    fn cast_same_ext_dtype() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);

        let output = arr
            .clone()
            .into_array()
            .cast(DType::Extension(ext_dtype.clone()))
            .unwrap();
        assert_eq!(arr.len(), output.len());
        assert_eq!(arr.dtype(), output.dtype());
        assert_eq!(output.dtype(), &DType::Extension(ext_dtype));
    }

    #[test]
    fn cast_same_ext_dtype_differet_nullability() {
        let ext_dtype = Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let storage = Buffer::<i64>::empty().into_array();

        let arr = ExtensionArray::new(ext_dtype.clone(), storage);
        assert!(!arr.dtype().is_nullable());

        let new_dtype = DType::Extension(ext_dtype).with_nullability(Nullability::Nullable);

        let output = arr.clone().into_array().cast(new_dtype.clone()).unwrap();
        assert_eq!(arr.len(), output.len());
        assert!(arr.dtype().eq_ignore_nullability(output.dtype()));
        assert_eq!(output.dtype(), &new_dtype);
    }

    #[test]
    fn cast_different_ext_dtype() {
        let original_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        // Note NS here instead of MS
        let target_dtype = Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased();

        let storage = buffer![1i64].into_array();
        let arr = ExtensionArray::new(original_dtype, storage);

        let result = arr
            .into_array()
            .cast(DType::Extension(target_dtype))
            .and_then(|a| {
                a.execute::<ExtensionArray>(&mut SESSION.create_execution_ctx())
                    .map(|c| c.into_array())
            });
        assert!(result.is_err());
    }

    #[test]
    fn cast_uses_source_extension_coercion() -> VortexResult<()> {
        let source_dtype = ExtDType::<MillisecondTimestamp>::try_new(
            EmptyMetadata,
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?
        .erased();
        let target_dtype =
            Timestamp::new(TimeUnit::Milliseconds, Nullability::NonNullable).erased();
        let source = ExtensionArray::new(source_dtype, buffer![1i64].into_array()).into_array();
        let target = DType::Extension(target_dtype);
        let incompatible_target = DType::Extension(
            Timestamp::new(TimeUnit::Nanoseconds, Nullability::NonNullable).erased(),
        );

        assert!(target.can_coerce_from(source.dtype()));
        assert!(source.dtype().can_coerce_to(&target));
        assert!(!source.dtype().can_coerce_from(&target));
        assert!(!target.can_coerce_to(source.dtype()));
        assert!(!incompatible_target.can_coerce_from(source.dtype()));
        let result = source
            .cast(target.clone())?
            .execute::<ExtensionArray>(&mut SESSION.create_execution_ctx())?;

        assert_eq!(result.dtype(), &target);
        Ok(())
    }

    #[test]
    fn cast_timestamp_to_i64() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let ext_dtype = Timestamp::new_with_tz(
            TimeUnit::Nanoseconds,
            Some("UTC".into()),
            Nullability::NonNullable,
        )
        .erased();
        let storage = buffer![1i64, 2, 3].into_array();
        let arr = ExtensionArray::new(ext_dtype, storage).into_array();

        let result = arr.cast(DType::Primitive(PType::I64, Nullability::NonNullable))?;
        assert_eq!(
            result.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_arrays_eq!(result, buffer![1i64, 2, 3].into_array(), &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(create_timestamp_array(TimeUnit::Milliseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Microseconds, true))]
    #[case(create_timestamp_array(TimeUnit::Nanoseconds, false))]
    #[case(create_timestamp_array(TimeUnit::Seconds, true))]
    fn test_cast_extension_conformance(#[case] array: ExtensionArray) {
        test_cast_conformance(&array.into_array(), &mut SESSION.create_execution_ctx());
    }

    fn create_timestamp_array(time_unit: TimeUnit, nullable: bool) -> ExtensionArray {
        let ext_dtype =
            Timestamp::new_with_tz(time_unit, Some("UTC".into()), nullable.into()).erased();

        let storage = if nullable {
            PrimitiveArray::from_option_iter([
                Some(1_000_000i64), // 1 second in microseconds
                None,
                Some(2_000_000),
                Some(3_000_000),
                None,
            ])
            .into_array()
        } else {
            buffer![1_000_000i64, 2_000_000, 3_000_000, 4_000_000, 5_000_000].into_array()
        };

        ExtensionArray::new(ext_dtype, storage)
    }
}
