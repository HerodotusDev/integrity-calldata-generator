#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// This function is called on panic.
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    loop {}
}

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[allow(unused_imports)]
use cairovm_verifier_air;
#[allow(unused_imports)]
use cairovm_verifier_commitment;
#[allow(unused_imports)]
use cairovm_verifier_pow;
#[allow(unused_imports)]
use cairovm_verifier_stark;
