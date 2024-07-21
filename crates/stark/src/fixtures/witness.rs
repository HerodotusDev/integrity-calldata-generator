use starknet_crypto::Felt;
use swiftness_commitment::{
    table::types::{Decommitment as TableDecommitment, Witness as TableCommitmentWitness},
    vector::types::Witness as VectorCommitmentWitness,
};

use crate::types::StarkWitness;

pub fn get() -> StarkWitness {
    StarkWitness {
        traces_decommitment: swiftness_air::fixtures::decommitment::get(),
        traces_witness: swiftness_air::fixtures::witness::get(),
        composition_decommitment: TableDecommitment {
            values: vec![
                Felt::from_hex_unchecked(
                    "0x7292f156e66e87304f8a7e0e565a3fec43d76f2c80a400c2a590c6660f205f0",
                ),
                Felt::from_hex_unchecked(
                    "0x1a5eb53e5336a81290ca99cbd8d3aa638bc94a830b39fdea42631fb79a8a18",
                ),
                Felt::from_hex_unchecked(
                    "0x635a6d597a5fa9e30719d9eb31fa71025849d5db13d52978f6c030ff83d5747",
                ),
                Felt::from_hex_unchecked(
                    "0x1b98ae2bcfe29abdff214a5d649d2b9b4861785e71114e4126a240bf88ea981",
                ),
                Felt::from_hex_unchecked(
                    "0x70bfef37afc339928c0b913331065c9f91de38950d4485ee3554b81049ca7a9",
                ),
                Felt::from_hex_unchecked(
                    "0x66530d11b117a33d3c4f625b53bd534209fce354190801eb5511d6e26e876f7",
                ),
                Felt::from_hex_unchecked(
                    "0x29c2c6fb927ec573b1b8d28eb0160e54b8386414ce096221c744924329a9b75",
                ),
                Felt::from_hex_unchecked(
                    "0x75913dbeef39e24469bc99ced5a8970e3c99a728979115a205198d928716059",
                ),
                Felt::from_hex_unchecked(
                    "0x764b5e049a3f2073449e18898e94b404f3c866e29dc24de116311c8ed9edf6c",
                ),
                Felt::from_hex_unchecked(
                    "0xfb84335e5da80feb729a0badcc7fc25152b599e8c760474422015ffffda485",
                ),
                Felt::from_hex_unchecked(
                    "0xb3aa6727ca035c9b754ba27b146e1aafae0b008cebf7e5c5ed30ea1ba3ef10",
                ),
                Felt::from_hex_unchecked(
                    "0x78bd952c31812a3dba9ea82034b8b0136c6e3259284678279222c2c7ec97528",
                ),
                Felt::from_hex_unchecked(
                    "0x2aabc8793bdd34d85d4bc8a65892e1a135711c8cf0cb3d28a686b3ce2258740",
                ),
                Felt::from_hex_unchecked(
                    "0x7cc04d09f944a52e78c4c0c3bf4c7b412332a63e4bdf6cbabaff72a2096092e",
                ),
                Felt::from_hex_unchecked(
                    "0x1e54ed7383142528172ca0a33fc6d858cabd2cb451dcf3e230415a554ab0904",
                ),
                Felt::from_hex_unchecked(
                    "0x1ee4666c435c870ad78576ff46d940d2a7c1ebbce56e478e27712b882383fa5",
                ),
                Felt::from_hex_unchecked(
                    "0x6b90ce41d9581f9cd6f59e3853d27731d1413e134bf4e460984d2069125444b",
                ),
                Felt::from_hex_unchecked(
                    "0x76033ecf0736a2871bd443651416ff0cadc0766d211e5664f3a0615e2934b5f",
                ),
                Felt::from_hex_unchecked(
                    "0x121f3f91e17d3331df67d77b6f81c288b538c382d1751d6f81b2fbe247c1470",
                ),
                Felt::from_hex_unchecked(
                    "0x18b1e78ed8bad85ab8eb8d5182462144bcce5da7385c14433a5f217b5bf7ef",
                ),
            ],
        },
        composition_witness: TableCommitmentWitness {
            vector: VectorCommitmentWitness {
                authentications: vec![
                    Felt::from_hex_unchecked(
                        "0x3cb594d08bff7d9b1358ff9ac9da944f817b8fdaf0ce0653ae610f164481017",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6beeb3913b3427a3593c2d86e6305000e58bf716d47efd3c2916d1920eebc42",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5f7769ba918cbb40a2828272e07d44ad7d3ce5c75ea4853d6217547f34039ae",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5122cc7612e6f47f2435a9b36e24c4d82ef3221cdc1fc5d58989a56765f87c1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x292e749ce842eecabd0145e88b667f8c8eb3454927618d066649f9029da01dd",
                    ),
                    Felt::from_hex_unchecked(
                        "0x675df17db3aa932063dc2582a6203d43ef047923db874043f74a663cda64cb",
                    ),
                    Felt::from_hex_unchecked(
                        "0x46f0094e0705f5db765eb17d841bf1dafe9267856aa8d9b9968a557855cd046",
                    ),
                    Felt::from_hex_unchecked(
                        "0xe4a16452992bbd753214dfb94769f538e6cfa7ad88934f696bd6b4e8072904",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5a8fecc7ea4a4937efac9693cdeb65b5aebd1e9b1f086a5f779c2d165d2695",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4895c07f340bcd35e4f586daea4d086a4ccdd1af31de176b9d3b4551d81ac34",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7efdbeae58a76eda09ab5c90a5770da9092b8317626d99ba2f05635e625c82d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x539b1eb49224bb04653719dd7da9779b155e957e98cdec817620f7e915bddf7",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1dfa81025ea1530060a750feaa27ba1d7104173de01f00562d09ad373393229",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2f8123a38cfcb85da8c1c17f376f0803d7942a992de9bd487fa3dc866d44a1c",
                    ),
                    Felt::from_hex_unchecked(
                        "0x78128f61d8039e60db9c9ef2bb889575997d64bf290b92aef145b9f707dcf03",
                    ),
                    Felt::from_hex_unchecked(
                        "0x146db43d42b40176c69bfc20bb5a6236a8af56211951c4ee2603de677e69f33",
                    ),
                    Felt::from_hex_unchecked(
                        "0x14a2311e0a67233639f98713fc97ba5484f45b88a9505a88caa7d2046504dd",
                    ),
                    Felt::from_hex_unchecked(
                        "0xbcdba8397236ae18a665525ab1beb3573818fec3cdc6bc9c000d08cbe4fd01",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1ee42a585b7dc319edc981d01e1e7d454eb331b04faea3340a3f0894b5fc0ba",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2af708156a134c9ee7e7458179c1f4a99188c75e014519ef09e697fc89646b3",
                    ),
                    Felt::from_hex_unchecked(
                        "0x67ff1e51c699a9e6609e8beef226d7cf87f0c0cb8697b42f1304f6d912402df",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7d69b2966bacced9b888d055c817576cc3934538922c9a87836f99383ac377b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6992840697dc922e5144c89a02532d7310bbb7c185d23f6b2899a1d362888c6",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1638a6e1a7c859f5cae7e04bd91bed8bf76d210ea61339a3d31becf05c54a57",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4b2f9b33be1c7117542863ec29818013b7aad58c827ac933a6a718d4b482df",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5d38fe2d23df29a6823a053154fdeaff1b2183bdeaa1e386fb7a0371be523f4",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7b48e0810cdc09560bdf33e85b359fb1b2e4f7a9fcdf6299a24259e3aa318d8",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4d329abf1e5174fba38df8ce481eae174bbd0aadb6a7a07404a265072242589",
                    ),
                    Felt::from_hex_unchecked(
                        "0xc10e250d68ed237dd66d9375485dbb597c112e91950fce43c41a2937434830",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3ff05ec7e2f91a2c152fce7e25fb00359ba73dd714e000face0323b0b9ad859",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6c42a1caa4e2daa10e36d39dd74dd0b47fc1a5fa86874ed7fdc3e3f8bfd9968",
                    ),
                    Felt::from_hex_unchecked(
                        "0x108233f77db3b22474d24c693f8c9f522f54936f0cf3277ac2c0cfb87645124",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2ab8a1eca87570c9b997e548f5ac722ade3cf9366d71c742befeb6a47e1cf0a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3c2affee148887742e134ea9ca11977a94b08679890b521f808ec7b2938a475",
                    ),
                    Felt::from_hex_unchecked(
                        "0x775753b51de5d58fd9bb81c9fad827c58d5dcc336153d0dac38ebb07e65f3f8",
                    ),
                    Felt::from_hex_unchecked(
                        "0xfe42ed357f99ab116d05b17fdabb3b6f6ee511fbb868fda7c699265165bd88",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1d5c3d5e0d84ddef1a622d374db1f8fc8135d338f563d95c751903f8f32141b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x38c163ca39f9196f79ff1a6c903e1df92840913f7a8547ab7f67feee58967ef",
                    ),
                    Felt::from_hex_unchecked(
                        "0x20600401d215deceb5cbc89919a5757357b09d67a2c75e72bf3e4fe067d0d0f",
                    ),
                    Felt::from_hex_unchecked(
                        "0x75007561b4962ca2fb8bba400509e2a3ee2a29090383c84d4abc2a5e4eacdad",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2c4673c0b845b0808ae989cf8bbda5c3c69c9f9548b60b465fbf36a9a28670a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1f8be87efb2b7b9421e75eb26a642edfd34069ee45dd3b932eceb252087b6a0",
                    ),
                    Felt::from_hex_unchecked(
                        "0x21543267189894009b51ef02befe0a0578abf1aaed28637f8d5c55407f8f528",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5fbc7f8c4fab96b1a4c4479417b7e386b389154a87872b341e85155c321565a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x349558f943bf50ccd7aa8beb3418c757f7214099cf74b3fc67ce6048d9bbbaf",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6463e1c0d14b29f21794a9d542a8a5404b8c81dfea5eea65c2c934375a98527",
                    ),
                    Felt::from_hex_unchecked(
                        "0x449570e44f82a84cb1e61bde26b1b513627f2d7ddcb86605516c3ecb86367fe",
                    ),
                    Felt::from_hex_unchecked(
                        "0x17b100ac783f096c1b72a51dd4d9301a52a6255c7c1187b7071ee391bef11d0",
                    ),
                    Felt::from_hex_unchecked(
                        "0x55ee5e95d057386481652bab3169514bc5ec69b965335eaf54eea5a50af7165",
                    ),
                    Felt::from_hex_unchecked(
                        "0x47856e9a05c92e36bce9b80b53469266f9d45f3b7e7445a3c17b134c983e375",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2bc36ba245abb63b03df27ceca6bee533a1adb32df006440e489f1d0688919a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7fe9d86cea470094d9ababab3b597a1f4a02784eac4330e36b6ad28a7a6bf08",
                    ),
                    Felt::from_hex_unchecked(
                        "0x18ff5439d87d54197ffb813a6fc877c321abf337234143b10defa65cd5a0833",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5fc5c99ef905ff0e3aa90d744a2ade01dc747417a8e65389b3e6dc960852109",
                    ),
                    Felt::from_hex_unchecked(
                        "0x38c8386123c89e94f12d22527b04f47736341fa55b384c0652ae3ce9ddc3e11",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7c35d50e96060aa4aba142437400323238167b3090941477d2bc6e84dc55823",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4deb3c05df15e01d3840051ec6d296923de78ed027211414a01691d038e5e3d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2c89cba9afc6416170e437a6d19acfd1b5064a6783cc5d86292d3d1c465e914",
                    ),
                    Felt::from_hex_unchecked(
                        "0x295cf13ff878e06be0e97c446b4f9741bdd032fa051c2c3157e5ee815374c66",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2f50486447607cd6f5166a1328befa62e4df1a4cc07c2e76ede1a1b8d0e2253",
                    ),
                    Felt::from_hex_unchecked(
                        "0x234f13465a7d4ab2c6180a1cbff778531dd61b95c6966f7ba7f695d5c5a16d3",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3918b3e434316b3522c1ea6722ccb562dc6364c0a6161540c6d6715cf40c351",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3fbfa8638a332a14721cc1285deacde81002291fcc95c5c8bd4b411c2a1bff2",
                    ),
                    Felt::from_hex_unchecked(
                        "0x39c4079c7394555e5a3397abd633d59808ecce390a37bc48affff0ddf10e933",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5024ab3aa4129429f863e34c76d209d7694a3aceffa3130afe5330b8064ee50",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6148d4955dd9a632a1f891447f7422edf7249a73db531d11c9404e052e9a66c",
                    ),
                    Felt::from_hex_unchecked(
                        "0x25e7ec740a351a2b6dcfb0f38d35fe46f11061920dd4684c71450dbe62dc172",
                    ),
                    Felt::from_hex_unchecked(
                        "0x60d30f218cb7054e5dcae4f5bfa204d33afecb8ff4ea81fb874faafa7825046",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5e1e68ff0aedbca5ef936bc41f339392be5b304c6d0b14368d5552a5f9846c0",
                    ),
                    Felt::from_hex_unchecked(
                        "0x69e738066df7e08a81dcab46b703c44bab1cffbe49f3412d38ddbe47ddab6a5",
                    ),
                    Felt::from_hex_unchecked(
                        "0x390fa4304de0305d750c3cd7895cec9ae78e0eb3149205eb12d174c89f2c71a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5fc3ebf3ed45eca26e6a0926ee47efa667384c16afa847f6994e21c5855d7ce",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5cd960acf717b64c434d60f381284280f5cacfcc24a06219de54341f5ee5d24",
                    ),
                    Felt::from_hex_unchecked(
                        "0x572f23cf3b699f71f7bc560ff23116d93aa820262164e803bdc8dc51cd54102",
                    ),
                    Felt::from_hex_unchecked(
                        "0x429445aa637f3e700e0852f30fded8b119fd37cb83a45fabb63f7c42ac4427d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x14809e472e3437aecff0296d38680979dd0c36623d36ce87602d6c4e561cfca",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3a8127e13d5a0b6f26fc5df086f1b58886c95536ec9b6414d7f24d201633d30",
                    ),
                    Felt::from_hex_unchecked(
                        "0x618c0228e7327fd5d226d347cea78f00cd371d247a0e89752af4827a4d30ca2",
                    ),
                    Felt::from_hex_unchecked(
                        "0x49a17c630c87ef4063e5d298be4bcf9dd10111c68ca9b7319a7c98d50509741",
                    ),
                    Felt::from_hex_unchecked(
                        "0x744d454bdb88286ab2750a2728a5263a1922b91de73e94480a364bbd704d12d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4af9fec824ec58ecf9a96f0de7658501586134d1034869c6a4b30bdd8494e8",
                    ),
                    Felt::from_hex_unchecked(
                        "0x25e5daf9778cb1d2bda48faa26abc633a3b6f49bccda55ca2f88fb77e4ff914",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2b1bfc7bf6e5d93263cacc437cc586fc47c16057e61037ecf1ff809fc7e2a60",
                    ),
                    Felt::from_hex_unchecked(
                        "0x13e040f6f42f426105b131487a29b10475c00572a41a6521fb8a7622ec198dc",
                    ),
                    Felt::from_hex_unchecked(
                        "0x631a61788f4355e59c5ebd1f66c25dd5a68c1a2bb54c356d1c8caae5102206f",
                    ),
                    Felt::from_hex_unchecked(
                        "0x30365d07d1e1741d10944d7ff7927ad0f80102169bdc0b587ef2184c98b4d0d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x30e12e014ae9072ef8e2e7af4bd208152b991f8bc2e2543d226fa1f38a2b4dd",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5fab63c6b62afa9b991bdbbc26cfa1f67ef6c7b4b840e49c257748ad5d35275",
                    ),
                    Felt::from_hex_unchecked(
                        "0x149d9b98f2612eea8e60d9c30aacc1a5de4228a271fdeccc8a6216b5c3e7b6e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7adce6bcecc82229d6a94fb56b8e73ddef84f0cc04d857a0058d663ced3be9e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x437dcc1721e94d6733d118c972c970b1a515da439fdef0a693d993ce758bee1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5460cb1e82124b8e8139c61ed5b868cb8d11be3aa76488cace76b10f5aba926",
                    ),
                    Felt::from_hex_unchecked(
                        "0x29a21f246546bb759ce21a496b9e540a517b3fd12d63903a882e5d157d31ad1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3b52205144f8e3c9f26cd99d30bda06d2152ffe6d24a220cc36c794d8127f7e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x72905e4e7d3ff5fff996e564e3009f7bb355a43ab7d26f6b89478ef45ed913a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5d2649cd1f03f9c16d42db2124a84a293cbbde2a95411fbbf30f2f66cb3387",
                    ),
                    Felt::from_hex_unchecked(
                        "0x62cd0e36757cb0cdd0aa74dedf0ec69b193470668944bed1fd017a32ef6948d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1ceb43411395fb6cbc9f1a2627431812b1fcdce67af91eec4f229382a181e9b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x86b5398d87497716f4b5fc41ec91c9870f519024ea18fdd827db9433a86afb",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4f677cab98dbfcf9bfb9862e17bcf2d3b6a45f1328ace805368d72bc06f20d1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1a051405e899273ba19e567ae833650933df5715c625a225e3f4ee77d6c9c",
                    ),
                    Felt::from_hex_unchecked(
                        "0x247f785474893aed6be3ee0249f6afa9557af30ebfe8e3b78867540ed8c0f4f",
                    ),
                    Felt::from_hex_unchecked(
                        "0x436c88dfaca7845283a3632581e04d935f2fbfc0cfb33c52a5caf3cc016f3ab",
                    ),
                    Felt::from_hex_unchecked(
                        "0x59cee8905d5819db41f373ff5bcfd80337e23de92938c30e01b6417ba7fdaf3",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6fc98023f05ff2b200033a575d561cdd46b0ef7fb343e4621848abd6d09359d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2d60fc4e54722740ec97a28008275aebbc26fded34f98186385274063cdcb8e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x25b1a4e6c3d23e63cd2d1192838f10efd5fe18bcf58a10092c5a7f775a3caa1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1864425ca686df29adcb4669f7fc6870c2d13c5c3b4c15d8e73f615ff66f47e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5507332cc546b5614039aee429309df762a286641f3002a0c51db7382630221",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6beea637e4bf5ea7d68fbc37e024e87db696a3cbfb79503f546d5adc4ade297",
                    ),
                    Felt::from_hex_unchecked(
                        "0x784b742ccdc4907653cee8cac006c440d964a1b617e12587829f685d7de957b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2daa54942cd0e22e26bf7a296462fc31726a1e63776aaec3898421555c7d6fc",
                    ),
                    Felt::from_hex_unchecked(
                        "0x23c26c28f00a5f3a0125883cf5442d74f94fe55279f9fafb6b0302a354e1c0c",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3b274eaec5bc0dca6554bf543631863294bc5582e59ed8441bde65c76d01819",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3ab74056d69acb0952862f1d42d3570593bcb207dcd2b2cd2f1725a8f2e11e5",
                    ),
                    Felt::from_hex_unchecked(
                        "0x738828c6509bd7c90f50e296dec71a6dac97158abbec9bd316d8fa8024dc402",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7cc1a27255d6d114a53fa68f0cc68929146ff9171772cb952c10871451deb7f",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6eb82b3ffffd8df6962241c1f6c2fccf5843e3e5fd6ceab73bed56ecfcd783f",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1f807c0f1b03d1c6285db6d1639b23dd78586df0b6adc2b620bfeca154644b1",
                    ),
                    Felt::from_hex_unchecked(
                        "0xe27f902b69f3df42e030038102847a91e48dce5cd08b207630f4ae70ecd8fc",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3db474f7a355d660850e85dc34865db983cbc73424b353b12f8f0058fe639d1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x574ad29fba6900188149192884996d9786de1c9b5f34e78b624137420e6ec7d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x25e31f25948dc259206c68fe3fc81da11f693772e5e1120043aec7f58d1432b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4bb5dbd41679ee8355c8ad6d3f0780be549631e398e23bc4d63bd89c201bc66",
                    ),
                    Felt::from_hex_unchecked(
                        "0x54f09e9cc0bea3eabb4b27bcfc5e68a6454273e00d99a6eb2dee3defca84043",
                    ),
                    Felt::from_hex_unchecked(
                        "0x637f2a47a217fcacce3231ab81927b5735c0fea75f1a6374c48f5b3b0e1e047",
                    ),
                    Felt::from_hex_unchecked(
                        "0x37cf863ca944c970eb00cc79e2459e4dbcd42e2200f9e88e0935645c9a9a025",
                    ),
                    Felt::from_hex_unchecked(
                        "0x281eeb25c656467ddb8fc383ee46bf85d02c70e2043415319724494f23eff04",
                    ),
                    Felt::from_hex_unchecked(
                        "0x473aba702a0fc32b73daeea042a7ef73aa50a252cc56995330d86fac854f27e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x7b39e7b9dbed77a62a05edb4b4092d9d1b3a2804ec567bb55cbdcb9a98a5886",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5e6864a1413eea78bd6f0b7d5435bdf27107caef1f0d75649cd5410c591d38c",
                    ),
                    Felt::from_hex_unchecked(
                        "0x36b1fb8a04a319fc19411ea7d332c2c0949a5ae0aa8b9afbf9869063f112b9",
                    ),
                    Felt::from_hex_unchecked(
                        "0x68442da2c45c834da3bc13f73663ded7d5b3177a0fd5a140db868489d66c1f1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x78e47a8b710b5b2c0430506ad9c237a8fc232468bc841e8d21f85d7cf492d8",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5f3445e52e9d18f913736d6041e19cc0fdee26aa536e7c2db566250f97fb0fa",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1f6286af5371648dcdd63e59d148c2aa7e310956e66244d6a05e6d6f51967d6",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6147b24028f88a6910126d98fa75161968c4d158fd138c6f188bcf62e25840d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x16205bb839ab37d3a59d7ecbde012de243a3bae36fc71cdd7e5e86fa09c22e1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2f37b53c53e463db1cec0df3c0ce90f6be7914c6b46bf987eb7a34729bbed70",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1f5455a1df27239cb2c1c0bd09aefc0a4398d98dea440019b3fb7c5ceea9d5e",
                    ),
                    Felt::from_hex_unchecked(
                        "0x380e5e05cac0a77b2867bc6bffdc8b0f79497ae0ac3335ef036b0de906dd06b",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6e27140b9954bf51cd66089e6c5b28df620a50515747449eb8b0b0532ea37b1",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2ede6732683021ca92415e3f1ff06639d4f849d2a72581e5311370edff63769",
                    ),
                    Felt::from_hex_unchecked(
                        "0x10ff3b0f7a9b94ac75d1ef55a6c655e17637d53ca9203dacf960fed7fe283e7",
                    ),
                    Felt::from_hex_unchecked(
                        "0x475cf06599139677f2b9539124512d8dd62ab20526e93d229252ceb4f5c9597",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1e0980b397c1624f781bb7fb706a03b1b3ae80a8ff0b0764442bbb9d93ecc1a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x580e8729a9024e0f4d600affab2449afab794f2213d036ba20e31d750b2be9a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x6554cc0f34173f2aa4c332cbf39bbd8f3bca537eca468195dad27d7d7b64129",
                    ),
                    Felt::from_hex_unchecked(
                        "0x1ae169fc73914d49b3e9b00c1c9a542cdca984830de97dfd9425260bff9e1ff",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3ac22604e3790bb579e53fc1c016b91ce13cfb7f9374cf28a7da7ca3e24f873",
                    ),
                    Felt::from_hex_unchecked(
                        "0x5bb9e0088c52577f6e378d15ec95633988f8e1dc6fc71c9bc4c78240bd79fb7",
                    ),
                    Felt::from_hex_unchecked(
                        "0x36411a1b391b64d0dafa34c3f1eb6451171f689df0405656f2091f560764e48",
                    ),
                    Felt::from_hex_unchecked(
                        "0x3d39b571382b4685b36631984acc33e3688c7e6ee17775d8a4f07e39ebe4987",
                    ),
                    Felt::from_hex_unchecked(
                        "0x18dfe63bef7aa584edb241c740d0de2efc2ef48d54fbd0e3983cd6aadbfa71d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x645d88db1e683befc0ed6014d277795a7ebd22fbef1ce620afee419e334889a",
                    ),
                    Felt::from_hex_unchecked(
                        "0x2e3c72530e78ac9c79e9bdf8ece9c1976d848a1c48b5922c2f2df1fac908a49",
                    ),
                    Felt::from_hex_unchecked(
                        "0x357e35aad82d57c34cbbd8b70df26fd403d0ee0198e6bf1d10218cfa5bb874d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x68fa850803cebb815a363eae319a6068e915ca61653a2672387b7e63c299a6d",
                    ),
                    Felt::from_hex_unchecked(
                        "0x4222a4a1d42276570fefd7dde277e64c2c064480d2bca726920fd9dcb27e755",
                    ),
                    Felt::from_hex_unchecked(
                        "0x52d152c87e6520b614e7851b8ee820b87aa9e1bcffa93f75191122822b9a1eb",
                    ),
                ],
            },
        },
        fri_witness: swiftness_fri::fixtures::witness::get(),
    }
}
