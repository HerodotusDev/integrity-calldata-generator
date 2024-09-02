pub mod autogenerated;
pub mod global_values;

use crate::{
    alloc::borrow::ToOwned,
    diluted::get_diluted_product,
    layout::stark_curve,
    periodic_columns::{
        eval_ecdsa_x, eval_ecdsa_y, eval_keccak_round_key0, eval_keccak_round_key1,
        eval_keccak_round_key15, eval_keccak_round_key3, eval_keccak_round_key31,
        eval_keccak_round_key63, eval_keccak_round_key7, eval_pedersen_x, eval_pedersen_y,
        eval_poseidon_poseidon_full_round_key0, eval_poseidon_poseidon_full_round_key1,
        eval_poseidon_poseidon_full_round_key2, eval_poseidon_poseidon_partial_round_key0,
        eval_poseidon_poseidon_partial_round_key1,
    },
    public_memory::{PublicInput, INITIAL_PC, MAX_ADDRESS, MAX_LOG_N_STEPS, MAX_RANGE_CHECK},
};
use alloc::vec;
use alloc::vec::Vec;
use global_values::{CurveConfig, EcPoint, EcdsaSigConfig, GlobalValues, InteractionElements};
use starknet_core::types::NonZeroFelt;
use starknet_crypto::{pedersen_hash, Felt};
use swiftness_commitment::table::{commit::table_commit, decommit::table_decommit};
use swiftness_transcript::ensure;

use super::{
    CompositionPolyEvalError, GenericLayoutTrait, LayoutTrait, OodsPolyEvalError, PublicInputError,
};

pub const N_DYNAMIC_PARAMS: usize = 340;
pub const N_CONSTRAINTS: usize = 419;
pub const MASK_SIZE: usize = 941;
pub const CPU_COMPONENT_HEIGHT: usize = 16;
pub const PUBLIC_MEMORY_FRACTION: usize = 8;
pub const HAS_DILUTED_POOL: usize = 1;
pub const DILUTED_SPACING: usize = 4;
pub const DILUTED_N_BITS: usize = 16;
pub const PEDERSEN_BUILTIN_REPETITIONS: usize = 1;
pub const RANGE_CHECK_N_PARTS: usize = 8;
pub const ECDSA_BUILTIN_REPETITIONS: usize = 1;
pub const ECDSA_ELEMENT_BITS: usize = 251;
pub const ECDSA_ELEMENT_HEIGHT: usize = 256;
pub const BITWISE_TOTAL_N_BITS: usize = 251;
pub const EC_OP_SCALAR_HEIGHT: usize = 256;
pub const EC_OP_N_BITS: usize = 252;
pub const POSEIDON_M: usize = 3;
pub const POSEIDON_ROUNDS_FULL: usize = 8;
pub const POSEIDON_ROUNDS_PARTIAL: usize = 83;
pub const RANGE_CHECK96_N_PARTS: usize = 6;
pub const ADD_MOD_WORD_BIT_LEN: usize = 96;
pub const ADD_MOD_N_WORDS: usize = 4;
pub const ADD_MOD_BATCH_SIZE: usize = 1;
pub const MUL_MOD_WORD_BIT_LEN: usize = 96;
pub const MUL_MOD_N_WORDS: usize = 4;
pub const MUL_MOD_BATCH_SIZE: usize = 1;
pub const HAS_OUTPUT_BUILTIN: usize = 1;
pub const HAS_PEDERSEN_BUILTIN: usize = 1;
pub const HAS_RANGE_CHECK_BUILTIN: usize = 1;
pub const HAS_ECDSA_BUILTIN: usize = 1;
pub const HAS_BITWISE_BUILTIN: usize = 1;
pub const HAS_EC_OP_BUILTIN: usize = 1;
pub const HAS_KECCAK_BUILTIN: usize = 1;
pub const HAS_POSEIDON_BUILTIN: usize = 1;
pub const HAS_RANGE_CHECK96_BUILTIN: usize = 1;
pub const HAS_ADD_MOD_BUILTIN: usize = 1;
pub const HAS_MUL_MOD_BUILTIN: usize = 1;
pub const LAYOUT_CODE: Felt = Felt::from_hex_unchecked("0x64796e616d6963");
pub const CONSTRAINT_DEGREE: usize = 2;
pub const IS_DYNAMIC_AIR: usize = 1;
pub const KECCAK_PERMUTATIONS_PER_INSTANCE: usize = DILUTED_N_BITS;

pub mod segments {
    pub const PROGRAM: usize = 0;
    pub const EXECUTION: usize = 1;
    pub const OUTPUT: usize = 2;
    pub const PEDERSEN: usize = 3;
    pub const RANGE_CHECK: usize = 4;
    pub const ECDSA: usize = 5;
    pub const BITWISE: usize = 6;
    pub const EC_OP: usize = 7;
    pub const KECCAK: usize = 8;
    pub const POSEIDON: usize = 9;
    pub const RANGE_CHECK96: usize = 10;
    pub const ADD_MOD: usize = 11;
    pub const MUL_MOD: usize = 12;
    pub const N_SEGMENTS: usize = 13;
}

pub mod builtins {
    use starknet_crypto::Felt;

    pub const OUTPUT: Felt = Felt::from_hex_unchecked("0x6F7574707574");
    pub const PEDERSEN: Felt = Felt::from_hex_unchecked("0x706564657273656E");
    pub const RANGE_CHECK: Felt = Felt::from_hex_unchecked("0x72616E67655F636865636B");
    pub const ECDSA: Felt = Felt::from_hex_unchecked("0x6563647361");
    pub const BITWISE: Felt = Felt::from_hex_unchecked("0x62697477697365");
    pub const EC_OP: Felt = Felt::from_hex_unchecked("0x65635F6F70");
    pub const KECCAK: Felt = Felt::from_hex_unchecked("0x6B656363616B");
    pub const POSEIDON: Felt = Felt::from_hex_unchecked("0x706F736569646F6E");
    pub const RANGE_CHECK96: Felt = Felt::from_hex_unchecked("0x72616E67655F636865636B3936");
    pub const ADD_MOD: Felt = Felt::from_hex_unchecked("0x6164645F6D6F64");
    pub const MUL_MOD: Felt = Felt::from_hex_unchecked("0x6D756C5F6D6F64");
}

// Pedersen builtin
pub const SHIFT_POINT_X: Felt =
    Felt::from_hex_unchecked("0x49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804");
pub const SHIFT_POINT_Y: Felt =
    Felt::from_hex_unchecked("0x3ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a");

pub const BUILTINS: [Felt; 11] = [
    builtins::OUTPUT,
    builtins::PEDERSEN,
    builtins::RANGE_CHECK,
    builtins::ECDSA,
    builtins::BITWISE,
    builtins::EC_OP,
    builtins::KECCAK,
    builtins::POSEIDON,
    builtins::RANGE_CHECK96,
    builtins::ADD_MOD,
    builtins::MUL_MOD,
];

pub struct Layout {}

impl GenericLayoutTrait for Layout {
    fn get_num_columns_first(public_input: &PublicInput) -> Option<usize> {
        public_input.dynamic_params.as_ref().map(|d| d.num_columns_first)
    }
    fn get_num_columns_second(public_input: &PublicInput) -> Option<usize> {
        public_input.dynamic_params.as_ref().map(|d| d.num_columns_second)
    }
}

impl LayoutTrait for Layout {
    const CONSTRAINT_DEGREE: usize = 2;
    const MASK_SIZE: usize = 941;
    const N_CONSTRAINTS: usize = 419;
    type InteractionElements = InteractionElements;

    fn eval_composition_polynomial(
        interaction_elements: &Self::InteractionElements,
        public_input: &PublicInput,
        mask_values: &[Felt],
        constraint_coefficients: &[Felt],
        point: &Felt,
        trace_domain_size: &Felt,
        trace_generator: &Felt,
    ) -> Result<Felt, CompositionPolyEvalError> {
        let dynamic_params = public_input
            .dynamic_params
            .to_owned()
            .ok_or(CompositionPolyEvalError::DynamicParamsMissing)?;
        let memory_z = interaction_elements.memory_multi_column_perm_perm_interaction_elm;
        let memory_alpha = interaction_elements.memory_multi_column_perm_hash_interaction_elm0;

        // Public memory
        let public_memory_column_size =
            trace_domain_size.field_div(&NonZeroFelt::from_felt_unchecked(
                Felt::from(dynamic_params.memory_units_row_ratio)
                    * Felt::from(PUBLIC_MEMORY_FRACTION),
            ));
        ensure!(
            public_memory_column_size < u128::MAX.into(),
            CompositionPolyEvalError::ValueOutOfRange
        );
        let public_memory_prod_ratio = public_input.get_public_memory_product_ratio(
            memory_z,
            memory_alpha,
            public_memory_column_size,
        );

        // Diluted
        let diluted_z = interaction_elements.diluted_check_interaction_z;
        let diluted_alpha = interaction_elements.diluted_check_interaction_alpha;
        let diluted_prod = get_diluted_product(
            DILUTED_N_BITS.into(),
            DILUTED_SPACING.into(),
            diluted_z,
            diluted_alpha,
        );

        // Periodic columns
        let (pedersen_points_x, pedersen_points_y) = if dynamic_params.uses_pedersen_builtin == 0 {
            (Felt::ZERO, Felt::ZERO)
        } else {
            let n_pedersen_hash_copies =
                trace_domain_size.field_div(&NonZeroFelt::from_felt_unchecked(
                    Felt::from(dynamic_params.pedersen_builtin_row_ratio)
                        * Felt::from(PEDERSEN_BUILTIN_REPETITIONS),
                ));
            ensure!(
                n_pedersen_hash_copies < u128::MAX.into(),
                CompositionPolyEvalError::ValueOutOfRange
            );
            let pedersen_point = point.pow_felt(&n_pedersen_hash_copies);
            (eval_pedersen_x(pedersen_point), eval_pedersen_y(pedersen_point))
        };

        let (ecdsa_generator_points_x, ecdsa_generator_points_y) =
            if dynamic_params.uses_ecdsa_builtin == 0 {
                (Felt::ZERO, Felt::ZERO)
            } else {
                let n_ecdsa_signature_copies =
                    trace_domain_size.field_div(&NonZeroFelt::from_felt_unchecked(
                        Felt::from(dynamic_params.ecdsa_builtin_row_ratio)
                            * Felt::from(ECDSA_BUILTIN_REPETITIONS),
                    ));
                ensure!(
                    n_ecdsa_signature_copies < u128::MAX.into(),
                    CompositionPolyEvalError::ValueOutOfRange
                );
                let ecdsa_point = point.pow_felt(&n_ecdsa_signature_copies);
                (eval_ecdsa_x(ecdsa_point), eval_ecdsa_y(ecdsa_point))
            };

        let (
            keccak_keccak_keccak_round_key0,
            keccak_keccak_keccak_round_key1,
            keccak_keccak_keccak_round_key3,
            keccak_keccak_keccak_round_key7,
            keccak_keccak_keccak_round_key15,
            keccak_keccak_keccak_round_key31,
            keccak_keccak_keccak_round_key63,
        ) = if dynamic_params.uses_keccak_builtin == 0 {
            (Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO)
        } else {
            let n_keccak_component_copies =
                trace_domain_size.field_div(&NonZeroFelt::from_felt_unchecked(
                    Felt::from(dynamic_params.keccak_row_ratio)
                        * Felt::from(KECCAK_PERMUTATIONS_PER_INSTANCE),
                ));
            ensure!(
                n_keccak_component_copies < u128::MAX.into(),
                CompositionPolyEvalError::ValueOutOfRange
            );
            let n_keccak_periodic_columns_copies = Felt::from(2048) * n_keccak_component_copies;
            let keccak_point = point.pow_felt(&n_keccak_periodic_columns_copies);
            (
                eval_keccak_round_key0(keccak_point),
                eval_keccak_round_key1(keccak_point),
                eval_keccak_round_key3(keccak_point),
                eval_keccak_round_key7(keccak_point),
                eval_keccak_round_key15(keccak_point),
                eval_keccak_round_key31(keccak_point),
                eval_keccak_round_key63(keccak_point),
            )
        };

        let (
            poseidon_poseidon_full_round_key0,
            poseidon_poseidon_full_round_key1,
            poseidon_poseidon_full_round_key2,
            poseidon_poseidon_partial_round_key0,
            poseidon_poseidon_partial_round_key1,
        ) = if dynamic_params.uses_poseidon_builtin == 0 {
            (Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO, Felt::ZERO)
        } else {
            let n_poseidon_copies = trace_domain_size.field_div(&NonZeroFelt::from_felt_unchecked(
                dynamic_params.poseidon_row_ratio.into(),
            ));
            ensure!(
                n_poseidon_copies < u128::MAX.into(),
                CompositionPolyEvalError::ValueOutOfRange
            );
            let poseidon_point = point.pow_felt(&n_poseidon_copies);
            (
                eval_poseidon_poseidon_full_round_key0(poseidon_point),
                eval_poseidon_poseidon_full_round_key1(poseidon_point),
                eval_poseidon_poseidon_full_round_key2(poseidon_point),
                eval_poseidon_poseidon_partial_round_key0(poseidon_point),
                eval_poseidon_poseidon_partial_round_key1(poseidon_point),
            )
        };

        let global_values = GlobalValues {
            trace_length: *trace_domain_size,
            initial_pc: public_input
                .segments
                .get(crate::layout::segments::PROGRAM)
                .ok_or(CompositionPolyEvalError::SegmentMissing {
                    segment: crate::layout::segments::PROGRAM,
                })?
                .begin_addr,
            final_pc: public_input
                .segments
                .get(crate::layout::segments::PROGRAM)
                .ok_or(CompositionPolyEvalError::SegmentMissing {
                    segment: crate::layout::segments::PROGRAM,
                })?
                .stop_ptr,
            initial_ap: public_input
                .segments
                .get(crate::layout::segments::EXECUTION)
                .ok_or(CompositionPolyEvalError::SegmentMissing {
                    segment: crate::layout::segments::EXECUTION,
                })?
                .begin_addr,
            final_ap: public_input
                .segments
                .get(crate::layout::segments::EXECUTION)
                .ok_or(CompositionPolyEvalError::SegmentMissing {
                    segment: crate::layout::segments::EXECUTION,
                })?
                .stop_ptr,
            initial_pedersen_addr: public_input
                .segments
                .get(segments::PEDERSEN)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::PEDERSEN })?
                .begin_addr,
            initial_range_check_addr: public_input
                .segments
                .get(segments::RANGE_CHECK)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::RANGE_CHECK })?
                .begin_addr,
            initial_ecdsa_addr: public_input
                .segments
                .get(segments::ECDSA)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::ECDSA })?
                .begin_addr,
            initial_bitwise_addr: public_input
                .segments
                .get(segments::BITWISE)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::BITWISE })?
                .begin_addr,
            initial_ec_op_addr: public_input
                .segments
                .get(segments::EC_OP)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::EC_OP })?
                .begin_addr,
            initial_keccak_addr: public_input
                .segments
                .get(segments::KECCAK)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::KECCAK })?
                .begin_addr,
            initial_poseidon_addr: public_input
                .segments
                .get(segments::POSEIDON)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::POSEIDON })?
                .begin_addr,
            initial_range_check96_addr: public_input
                .segments
                .get(segments::RANGE_CHECK96)
                .ok_or(CompositionPolyEvalError::SegmentMissing {
                    segment: segments::RANGE_CHECK96,
                })?
                .begin_addr,
            add_mod_initial_mod_addr: public_input
                .segments
                .get(segments::ADD_MOD)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::ADD_MOD })?
                .begin_addr,
            mul_mod_initial_mod_addr: public_input
                .segments
                .get(segments::MUL_MOD)
                .ok_or(CompositionPolyEvalError::SegmentMissing { segment: segments::MUL_MOD })?
                .begin_addr,
            range_check_min: public_input.range_check_min,
            range_check_max: public_input.range_check_max,
            offset_size: Felt::from(0x10000),     // 2**16
            half_offset_size: Felt::from(0x8000), // 2**15
            pedersen_shift_point: EcPoint { x: SHIFT_POINT_X, y: SHIFT_POINT_Y },
            ecdsa_sig_config: EcdsaSigConfig {
                alpha: stark_curve::ALPHA,
                beta: stark_curve::BETA,
                shift_point: EcPoint { x: SHIFT_POINT_X, y: SHIFT_POINT_Y },
            },
            ec_op_curve_config: CurveConfig { alpha: stark_curve::ALPHA, beta: stark_curve::BETA },
            pedersen_points_x,
            pedersen_points_y,
            ecdsa_generator_points_x,
            ecdsa_generator_points_y,
            keccak_keccak_keccak_round_key0,
            keccak_keccak_keccak_round_key1,
            keccak_keccak_keccak_round_key3,
            keccak_keccak_keccak_round_key7,
            keccak_keccak_keccak_round_key15,
            keccak_keccak_keccak_round_key31,
            keccak_keccak_keccak_round_key63,
            poseidon_poseidon_full_round_key0,
            poseidon_poseidon_full_round_key1,
            poseidon_poseidon_full_round_key2,
            poseidon_poseidon_partial_round_key0,
            poseidon_poseidon_partial_round_key1,
            memory_multi_column_perm_perm_interaction_elm: memory_z,
            memory_multi_column_perm_hash_interaction_elm0: memory_alpha,
            range_check16_perm_interaction_elm: interaction_elements
                .range_check16_perm_interaction_elm,
            diluted_check_permutation_interaction_elm: interaction_elements
                .diluted_check_permutation_interaction_elm,
            diluted_check_interaction_z: diluted_z,
            diluted_check_interaction_alpha: diluted_alpha,
            add_mod_interaction_elm: interaction_elements.add_mod_interaction_elm,
            mul_mod_interaction_elm: interaction_elements.mul_mod_interaction_elm,
            memory_multi_column_perm_perm_public_memory_prod: public_memory_prod_ratio,
            range_check16_perm_public_memory_prod: Felt::ONE,
            diluted_check_first_elm: Felt::ZERO,
            diluted_check_permutation_public_memory_prod: Felt::ONE,
            diluted_check_final_cum_val: diluted_prod,
        };

        autogenerated::eval_composition_polynomial_inner(
            mask_values,
            constraint_coefficients,
            point,
            trace_generator,
            &global_values,
            &dynamic_params,
        )
    }
    fn eval_oods_polynomial(
        public_input: &PublicInput,
        column_values: &[Felt],
        oods_values: &[Felt],
        constraint_coefficients: &[Felt],
        point: &Felt,
        oods_point: &Felt,
        trace_generator: &Felt,
    ) -> Result<Felt, OodsPolyEvalError> {
        let dynamic_params = public_input
            .dynamic_params
            .to_owned()
            .ok_or(OodsPolyEvalError::DynamicParamsMissing)?;
        autogenerated::eval_oods_polynomial_inner::<Self>(
            column_values,
            oods_values,
            constraint_coefficients,
            point,
            oods_point,
            trace_generator,
            &dynamic_params,
        )
    }
    fn traces_commit(
        transcript: &mut swiftness_transcript::transcript::Transcript,
        unsent_commitment: &crate::trace::UnsentCommitment,
        config: crate::trace::config::Config,
    ) -> crate::trace::Commitment<Self::InteractionElements> {
        // Read original commitment.
        let original_commitment =
            table_commit(transcript, unsent_commitment.original, config.original);

        // Generate interaction elements for the first interaction.
        let interaction_elements = Self::InteractionElements::new(transcript);

        // Read interaction commitment.
        let interaction_commitment =
            table_commit(transcript, unsent_commitment.interaction, config.interaction);

        crate::trace::Commitment {
            original: original_commitment,
            interaction_elements,
            interaction: interaction_commitment,
        }
    }
    fn traces_decommit(
        queries: &[Felt],
        commitment: crate::trace::Commitment<Self::InteractionElements>,
        decommitment: crate::trace::Decommitment,
        witness: crate::trace::Witness,
    ) -> Result<(), crate::trace::decommit::Error> {
        Ok(table_decommit(commitment.original, queries, decommitment.original, witness.original)
            .and(table_decommit(
                commitment.interaction,
                queries,
                decommitment.interaction,
                witness.interaction,
            ))?)
    }
    fn validate_public_input(
        public_input: &PublicInput,
        stark_domains: &crate::domains::StarkDomains,
    ) -> Result<(), PublicInputError> {
        let dynamic_params =
            public_input.dynamic_params.to_owned().ok_or(PublicInputError::DynamicParamsMissing)?;
        ensure!(public_input.log_n_steps < MAX_LOG_N_STEPS, PublicInputError::MaxSteps);

        let n_steps = Felt::TWO.pow_felt(&public_input.log_n_steps);
        let trace_length = stark_domains.trace_domain_size;
        ensure!(
            n_steps
                * Felt::from(CPU_COMPONENT_HEIGHT)
                * Felt::from(dynamic_params.cpu_component_step)
                == trace_length,
            PublicInputError::TraceLengthInvalid
        );

        ensure!(Felt::ZERO <= public_input.range_check_min, PublicInputError::RangeCheckInvalid);
        ensure!(
            public_input.range_check_min < public_input.range_check_max,
            PublicInputError::RangeCheckInvalid
        );
        ensure!(
            public_input.range_check_max <= MAX_RANGE_CHECK,
            PublicInputError::RangeCheckInvalid
        );

        ensure!(public_input.layout == LAYOUT_CODE, PublicInputError::LayoutCodeInvalid);

        let output_uses = public_input
            .segments
            .get(crate::layout::segments::OUTPUT)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::OUTPUT })?
            .stop_ptr
            - public_input
                .segments
                .get(crate::layout::segments::OUTPUT)
                .ok_or(PublicInputError::SegmentMissing {
                    segment: crate::layout::segments::OUTPUT,
                })?
                .begin_addr;
        ensure!(output_uses <= u128::MAX.into(), PublicInputError::UsesInvalid);

        let pedersen_copies = if dynamic_params.uses_pedersen_builtin == 0 {
            Felt::ZERO
        } else {
            trace_length.field_div(&NonZeroFelt::from_felt_unchecked(Felt::from(
                dynamic_params.pedersen_builtin_row_ratio,
            )))
        };
        let pedersen_uses = (public_input
            .segments
            .get(segments::PEDERSEN)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::OUTPUT })?
            .stop_ptr
            - public_input
                .segments
                .get(segments::PEDERSEN)
                .ok_or(PublicInputError::SegmentMissing {
                    segment: crate::layout::segments::OUTPUT,
                })?
                .begin_addr)
            .field_div(&NonZeroFelt::from_felt_unchecked(Felt::from(3)));
        ensure!(pedersen_uses <= pedersen_copies, PublicInputError::UsesInvalid);

        let range_check_copies = if dynamic_params.uses_range_check_builtin == 0 {
            Felt::ZERO
        } else {
            trace_length.field_div(&NonZeroFelt::from_felt_unchecked(Felt::from(
                dynamic_params.range_check_builtin_row_ratio,
            )))
        };
        let range_check_uses = public_input
            .segments
            .get(segments::RANGE_CHECK)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::OUTPUT })?
            .stop_ptr
            - public_input
                .segments
                .get(segments::RANGE_CHECK)
                .ok_or(PublicInputError::SegmentMissing {
                    segment: crate::layout::segments::OUTPUT,
                })?
                .begin_addr;
        ensure!(range_check_uses <= range_check_copies, PublicInputError::UsesInvalid);

        let bitwise_copies = if dynamic_params.uses_bitwise_builtin == 0 {
            Felt::ZERO
        } else {
            trace_length.field_div(&NonZeroFelt::from_felt_unchecked(Felt::from(
                dynamic_params.bitwise_row_ratio,
            )))
        };
        let bitwise_uses = (public_input
            .segments
            .get(segments::BITWISE)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::OUTPUT })?
            .stop_ptr
            - public_input
                .segments
                .get(segments::BITWISE)
                .ok_or(PublicInputError::SegmentMissing {
                    segment: crate::layout::segments::OUTPUT,
                })?
                .begin_addr)
            .field_div(&NonZeroFelt::from_felt_unchecked(Felt::from(5)));
        ensure!(bitwise_uses <= bitwise_copies, PublicInputError::UsesInvalid);

        Ok(())
    }

    fn verify_public_input(public_input: &PublicInput) -> Result<(Felt, Felt), PublicInputError> {
        let public_segments = &public_input.segments;

        let initial_pc = public_segments
            .get(crate::layout::segments::PROGRAM)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .begin_addr;
        let final_pc = public_segments
            .get(crate::layout::segments::PROGRAM)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .stop_ptr;
        let initial_ap = public_segments
            .get(crate::layout::segments::EXECUTION)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .begin_addr;
        let initial_fp = initial_ap;
        let final_ap = public_segments
            .get(crate::layout::segments::EXECUTION)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .stop_ptr;
        let output_start = public_segments
            .get(crate::layout::segments::OUTPUT)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .begin_addr;
        let output_stop = public_segments
            .get(crate::layout::segments::OUTPUT)
            .ok_or(PublicInputError::SegmentMissing { segment: crate::layout::segments::PROGRAM })?
            .stop_ptr;

        ensure!(initial_ap < MAX_ADDRESS, PublicInputError::MaxSteps);
        ensure!(final_ap < MAX_ADDRESS, PublicInputError::MaxSteps);

        // TODO support more pages?
        ensure!(public_input.continuous_page_headers.is_empty(), PublicInputError::MaxSteps);

        let memory = &public_input
            .main_page
            .iter()
            .flat_map(|v| vec![v.address, v.value])
            .collect::<Vec<Felt>>();

        ensure!(initial_pc == INITIAL_PC, PublicInputError::MaxSteps);
        ensure!(final_pc == INITIAL_PC + 4, PublicInputError::MaxSteps);

        let program_end_pc = initial_fp - 2;

        let program: Vec<&Felt> = memory
            .iter()
            .skip(initial_pc.to_bigint().try_into().unwrap())
            .step_by(2)
            .take((program_end_pc - Felt::ONE).to_bigint().try_into().unwrap())
            .collect();

        let hash = program.iter().fold(Felt::ZERO, |acc, &e| pedersen_hash(&acc, e));
        let program_hash = pedersen_hash(&hash, &Felt::from(program.len()));

        let output_len: usize = (output_stop - output_start).to_bigint().try_into().unwrap();
        let output = &memory[memory.len() - output_len * 2..];
        let hash =
            output.iter().skip(1).step_by(2).fold(Felt::ZERO, |acc, e| pedersen_hash(&acc, e));
        let output_hash = pedersen_hash(&hash, &Felt::from(output_len));

        Ok((program_hash, output_hash))
    }
}
