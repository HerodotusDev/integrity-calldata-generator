#[cfg(feature = "dex")]
use cairovm_verifier_air::layout::dex::Layout;
#[cfg(feature = "recursive")]
use cairovm_verifier_air::layout::recursive::Layout;
#[cfg(feature = "recursive_with_poseidon")]
use cairovm_verifier_air::layout::recursive_with_poseidon::Layout;
#[cfg(feature = "small")]
use cairovm_verifier_air::layout::small::Layout;
#[cfg(feature = "starknet")]
use cairovm_verifier_air::layout::starknet::Layout;
#[cfg(feature = "starknet_with_keccak")]
use cairovm_verifier_air::layout::starknet_with_keccak::Layout;

use cairovm_verifier_proof_parser::parse;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn cairovm_verify(proof: JsValue) -> Result<JsValue, JsError> {
    // Deserialize input from JsValue to Rust types
    let proof_str: String = proof.as_string().ok_or_else(|| JsError::new("Invalid input"))?;

    // Parse the proof
    let stark_proof = parse(proof_str).map_err(|e| JsError::new(&e.to_string()))?;

    // Get security bits and verify
    let security_bits = stark_proof.config.security_bits();
    let (program_hash, output_hash) = stark_proof
        .verify::<Layout>(security_bits)
        .map_err(|e| JsError::new(&e.to_string()))?;

    // Serialize result to JsValue
    let result = serde_json::to_value((program_hash, output_hash))
        .map_err(|e| JsError::new(&e.to_string()))?;
    Ok(JsValue::from_str(&result.to_string()))
}