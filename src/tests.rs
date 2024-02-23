use crate::utils::{bucket_hashes, hamming_distance, hash_vector};
use std::collections::HashMap;

#[test]
fn test_hash_vector() {
	let v = vec![2.3, 4.5, 6f64, 7.2];
	assert_eq!(hash_vector(v, 4), "0010".to_string());
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

#[test]
fn test_hamming_distance_panic() {
	assert!(hamming_distance("0010".to_string(), "100".to_string()).is_err());
}

#[test]
fn test_hamming_distance() {
	assert_eq!(hamming_distance("0010".to_string(), "1001".to_string()).unwrap(), 3u16);
}
