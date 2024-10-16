use eyre::bail;
use ndarray::Array1;
use openai::embeddings::{Embedding, Embeddings};
use std::{collections::HashMap, iter::zip};

type Hyperplanes =
	ndarray::prelude::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::prelude::Dim<[usize; 2]>>;

/// NOTE: This data file has around 570k text reviews (of types: single line, paragraph).
/// So, parse accordingly depending on the computational resources for bucketing.
pub(crate) const DATA_FILE: &str = "./data/fine_food_reviews_1k.csv";

/// subspace address format prefix as seed for hyperplane generation
pub(crate) const SEED: u64 = 2254;

fn update_text(text: &String) -> String {
	text.replace("\n", " ").replace("<br />", " ")
}

/// Get the embedding vector of a vec of texts with default OpenAI embedding small model.
/// Small embedding model: 1536 len of float values.
/// Large embedding model: 3072 len of float values.
pub(crate) async fn get_embeddings(text_samples: Vec<String>, model: &str) -> Vec<Embedding> {
	dotenv::dotenv().ok();
	openai::set_key(std::env::var("OPENAI_API_KEY").expect("Provide OpenAI API key?"));

	let owned_text_samples = text_samples.iter().map(|text| update_text(text)).collect::<Vec<_>>();

	// Then, create a Vec<&str> from the owned strings `Vec<String>`
	let text_samples_refs = owned_text_samples.iter().map(AsRef::as_ref).collect::<Vec<&str>>();

	// create embeddings
	Embeddings::create(model, text_samples_refs, "").await.unwrap().data
}

/// LSH random projection hash function with seeded hyperplane generation.
pub(crate) fn hash_vector(v: Vec<f64>, hyperplanes: Hyperplanes) -> String {
	// Convert Vec<Float64Type> to Array1<f64>
	let v_array = Array1::from_vec(v);

	// Dot product and thresholding to generate binary hash
	// Explicitly specify the expected type of `v_dot` as Array1<f64>
	let v_dot = hyperplanes.dot(&v_array);
	let binary_hash: Vec<u8> = v_dot.mapv(|x| if x > 0.0 { 1 } else { 0 }).to_vec();

	// Convert binary vector to hash string
	let hash_str = binary_hash.iter().map(|&x| x.to_string()).collect::<String>();
	hash_str
}

/// Distribute hashes into corresponding buckets
pub(crate) fn bucket_hashes(v: Vec<String>) -> HashMap<String, Vec<u128>> {
	let mut buckets = HashMap::new();

	for (i, hash_str) in v.iter().enumerate() {
		// Convert &str to String for the key
		// let key = hash_str.to_string();

		// Check if the key exists, and if not, insert a new vector
		// Then, always push the index (as u128, ensuring it fits into u128)
		buckets.entry(hash_str.to_owned()).or_insert_with(Vec::new).push(i as u128);
	}

	buckets
}

/// Calculate the Hamming distance between two strings.
pub(crate) fn hamming_distance(str1: String, str2: String) -> eyre::Result<u16> {
	if str1.len() != str2.len() {
		bail!("Strings must be of equal length");
	}

	let mut distance = 0u16;
	for (char1, char2) in zip(str1.chars(), str2.chars()) {
		if char1 != char2 {
			distance += 1;
		}
	}

	Ok(distance)
}
