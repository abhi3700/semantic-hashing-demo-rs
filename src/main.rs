use crate::utils::{DATA_FILE, SEED};
use ndarray::Array2;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use polars::prelude::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;

pub mod utils;
use utils::{bucket_hashes, get_embeddings, hash_vector};

#[cfg(test)]
mod tests;

fn collect_inputs() -> eyre::Result<(String, String, String, String)> {
	// load the .env file
	dotenv::dotenv().ok();
	dotenv::from_path("./.env").expect("Failed to get the .env file");

	let nbits = std::env::var("nbits").expect("Provide nbits?");
	let n = std::env::var("n").expect("Provide sample size?");
	let model = std::env::var("model").expect("Provide embedding model?");
	let query = std::env::var("query").expect("Provide query text?");

	Ok((nbits, n, model, query))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
	env_logger::init();

	let (nbits, n, model, query) = collect_inputs()?;

	// =============== A. Bucketing of Texts ===============
	println!("Parsing {} samples with {} hyperplanes for bucketing...", n, nbits);

	// knowledge base
	let df = CsvReader::from_path(DATA_FILE)?.infer_schema(None).finish()?;

	// Select the "Text" column
	let text_series = df.column("Text")?;

	// Convert the Series into a Vec<String>
	let mut reviews_vec: Vec<String> = text_series
		.str()?
		.into_iter()
		.filter_map(|opt| opt.map(|s| s.to_string()))
		.collect();

	// Take 1st 'n' reviews as samples
	reviews_vec.truncate(n.parse::<usize>()?);
	log::info!("\nInformation or Knowledge base (1st 5 samples):\n");
	for review_txt in reviews_vec.iter().take(5) {
		log::info!("{}", review_txt);
	}

	// If needed, convert Vec<String> to a flat ndarray
	// This step depends on your use case, as handling data in Vec<String> might be sufficient
	// let reviews_array = Array::from(reviews_vec);

	let embeddings = get_embeddings(reviews_vec, &model).await;
	log::debug!("1st embedding: {:?}", &embeddings[0].vec);
	log::debug!("and its len: {}", &embeddings[0].vec.len());

	// Create a seeded random number generator
	let mut rng = Pcg64::seed_from_u64(SEED);

	// Generate hyperplanes
	// FIXME: probably inside `random_using()`, arg-1, 2 are same as tested
	let hyperplanes: ndarray::prelude::ArrayBase<
		ndarray::OwnedRepr<f64>,
		ndarray::prelude::Dim<[usize; 2]>,
	> = Array2::random_using(
		(nbits.parse::<usize>()?, embeddings[0].vec.len()),
		Uniform::new(-0.5, 0.5),
		&mut rng,
	);

	// hash the embeddings vector
	let hashed_vectors = embeddings
		.iter()
		.map(|e| hash_vector((*e.vec).to_vec(), hyperplanes.clone()))
		.collect::<Vec<String>>();
	log::info!("The hashed vectors for all the samples:\n {:?}", hashed_vectors);

	// bucket the hashes into different buckets based on their hashes
	let buckets = bucket_hashes(hashed_vectors);
	println!("\nTotal no. of buckets is {} and they are:\n{:?}", buckets.keys().len(), buckets);

	Ok(())
}
