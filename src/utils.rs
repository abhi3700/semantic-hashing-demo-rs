use crate::constants::SEED;
use ndarray::{Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Get the embedding vector of a given text with default OpenAI embedding small model.
/// Small embedding model: 1536 len of float values.
/// Large embedding model: 3072 len of float values.
fn get_embedding(text: String, model: &str) {
	let text = text.replace('\n', " ");

	// TODO: create embedding
}

/// LSH random projection hash function with seeded hyperplane generation.
fn hash_vector(v: Vec<f64>, nbits: u16) -> String {
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
