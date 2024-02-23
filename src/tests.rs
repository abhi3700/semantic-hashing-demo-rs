use crate::utils::{bucket_hashes, hamming_distance, hash_vector, SEED};
use ndarray::Array2;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::HashMap;

#[test]
fn test_hash_vector() {
	// Create a seeded random number generator
	let mut rng = Pcg64::seed_from_u64(SEED);
	let nbits = 4;

	// Generate hyperplanes
	let hyperplanes: ndarray::prelude::ArrayBase<
		ndarray::OwnedRepr<f64>,
		ndarray::prelude::Dim<[usize; 2]>,
	> = Array2::random_using((nbits, nbits), Uniform::new(-0.5, 0.5), &mut rng);

	let v = vec![2.3, 4.5, 6f64, 7.2];
	assert_eq!(hash_vector(v, hyperplanes), "0010".to_string());
}

#[test]
fn test_bucket_hashes() {
	let hashed_vector = vec![
		"1001".to_string(),
		"1100".to_string(),
		"1101".to_string(),
		"1001".to_string(),
		"1110".to_string(),
		"1110".to_string(),
		"1100".to_string(),
		"1100".to_string(),
		"1101".to_string(),
		"1000".to_string(),
		"1111".to_string(),
		"1111".to_string(),
		"1101".to_string(),
		"1100".to_string(),
		"0100".to_string(),
		"0100".to_string(),
		"1111".to_string(),
		"0100".to_string(),
		"0010".to_string(),
		"0100".to_string(),
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

	let result: HashMap<String, Vec<u128>> =
		bucket_hashes(hashed_vector.iter().map(|s| s.to_owned()).collect());
	assert_eq!(result, expected);
}

#[test]
fn test_hamming_distance_panic() {
	assert!(hamming_distance("0010".to_string(), "100".to_string()).is_err());
}

#[test]
fn test_hamming_distance() {
	assert_eq!(hamming_distance("0010".to_string(), "1001".to_string()).unwrap(), 3u16);
}
