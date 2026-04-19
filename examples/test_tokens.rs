// Quick tokenizer parity diagnostic.
use rocmforge::loader::GgufFile;
use rocmforge::tokenizer::BpeTokenizer;

fn main() {
    let path = std::env::args().nth(1).expect("usage: test_tokens <gguf> [text]");
    let text = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "\n\nYou are a helpful assistant.".to_string());
    let file = GgufFile::open(&path).expect("open");
    let tok = BpeTokenizer::from_gguf(file.tokenizer_data());
    let ids = tok.encode(&text, true);
    for id in ids {
        let decoded = tok.decode(&[id], false);
        println!("{:6} -> {:?}", id, decoded);
    }
}
