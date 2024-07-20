# Cairo-VM Verifier

![Version](https://img.shields.io/badge/v0.0.3-green?style=flat-square&logo=git&logoColor=white&label=version)
![Continuous Integration](https://img.shields.io/github/actions/workflow/status/iosis-tech/cairovm-verifier/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=Continuous%20Integration)

This is the Rust implementation of the Cairo-VM STARK verifier with layouts. The code is inspired by StarkWare's [Cairo-verifier](https://github.com/starkware-libs/cairo-lang) implementation in Cairo0.

### Example usage
```js
import init, { cairovm_verify } from 'cairovm_verifier';

async function run(proof_json) {
  await init();  // Initialize the Wasm module
  try {
    const [programHash, programOutput] = JSON.parse( await cairovm_verify(proof_json) );
  } catch (err) {
    console.error(`Verification failed: ${err}`);
  }
}
```

## Contributing

Feel free to open issues or submit pull requests to help improve this project.