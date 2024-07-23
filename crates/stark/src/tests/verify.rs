use crate::{
    fixtures::{commitment, domains, witness},
    verify::stark_verify,
};
use swiftness_air::layout::{recursive::Layout, LayoutTrait};
use swiftness_fri::fixtures::queries;

#[test]
pub fn test_stark_verify() {
    let queries = queries::get();
    let commitment = commitment::get();
    let witness = witness::get();
    let stark_domains = domains::get();

    stark_verify::<Layout>(
        Layout::NUM_COLUMNS_FIRST,
        Layout::NUM_COLUMNS_SECOND,
        &queries,
        commitment,
        &witness,
        &stark_domains,
    )
    .unwrap()
}
