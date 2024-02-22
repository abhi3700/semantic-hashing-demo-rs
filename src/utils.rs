use crate::constants::SEED;
use eyre::bail;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::{collections::HashMap, iter::zip};

/// Get the embedding vector of a given text with default OpenAI embedding small model.
/// Small embedding model: 1536 len of float values.
/// Large embedding model: 3072 len of float values.
pub(crate) fn get_embedding(text: String, model: &str) {
	let text = text.replace('\n', " ");

	// TODO: create embedding
}

/// LSH random projection hash function with seeded hyperplane generation.
pub(crate) fn hash_vector(v: Vec<f64>, nbits: u16) -> String {
	// Convert Vec<Float64Type> to Array1<f64>
	let v_array = Array1::from_vec(v);

	// Create a seeded random number generator
	let mut rng = Pcg64::seed_from_u64(SEED);

	// Generate hyperplanes
	let hyperplanes =
		Array2::random_using((nbits as usize, v_array.len()), Uniform::new(-0.5, 0.5), &mut rng);

	// Dot product and thresholding to generate binary hash
	// Explicitly specify the expected type of `v_dot` as Array1<f64>
	let v_dot = hyperplanes.dot(&v_array);
	let binary_hash: Vec<u8> = v_dot.mapv(|x| if x > 0.0 { 1 } else { 0 }).to_vec();

	// Convert binary vector to hash string
	let hash_str = binary_hash.iter().map(|&x| x.to_string()).collect::<String>();
	hash_str
}

#[test]
fn test_hash_vector() {
	let v = vec![2.3, 4.5, 6f64, 7.2];
	assert_eq!(hash_vector(v, 4), "0010".to_string());
}

/// Distribute hashes into corresponding buckets
pub(crate) fn bucket_hashes(v: Vec<&str>) -> HashMap<String, Vec<u128>> {
	let mut buckets = HashMap::new();

	for (i, &hash_str) in v.iter().enumerate() {
		// Convert &str to String for the key
		let key = hash_str.to_string();

		// Check if the key exists, and if not, insert a new vector
		// Then, always push the index (as u128, ensuring it fits into u128)
		buckets.entry(key).or_insert_with(Vec::new).push(i as u128);
	}

	buckets
}

#[test]
fn test_bucket_hashes() {
	let hashed_vector = vec![
		"1001", "1100", "1101", "1001", "1110", "1110", "1100", "1100", "1101", "1000", "1111",
		"1111", "1101", "1100", "0100", "0100", "1111", "0100", "0010", "0100",
	];
	let mut expected = HashMap::new();
	expected.insert("1001".to_string(), vec![0, 3]);
	expected.insert("1100".to_string(), vec![1, 6, 7, 13]);
	expected.insert("1101".to_string(), vec![2, 8, 12]);
	expected.insert("1110".to_string(), vec![4, 5]);
	expected.insert("1000".to_string(), vec![9]);
	expected.insert("1111".to_string(), vec![10, 11, 16]);
	expected.insert("0100".to_string(), vec![14, 15, 17, 19]);
	expected.insert("0010".to_string(), vec![18]);

	let result = bucket_hashes(hashed_vector.iter().map(|&s| s).collect());
	assert_eq!(result, expected);
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

#[test]
fn test_hamming_distance_panic() {
	assert!(hamming_distance("0010".to_string(), "100".to_string()).is_err());
}

#[test]
fn test_hamming_distance() {
	assert_eq!(hamming_distance("0010".to_string(), "1001".to_string()).unwrap(), 3u16);
}
