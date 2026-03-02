use pyo3::prelude::*;
use std::collections::HashMap;

//accept retry thresholds determined by tuning during development

const ACCEPT_THRESHOLD: f64 = 0.55;
const RETRY_THRESHOLD:  f64 = 0.38;

// fusion weights determined by tuning during development
const W_SPATIAL: f64 = 0.45;  // ResNet18 max softmax prob
const W_TEXTURE: f64 = 0.30;  // Laplacian variance
const W_MOTION:  f64 = 0.25;  // inter-frame differencing
const W_DEPTH:   f64 = 0.00;  // MiDaS depth 
//weight is currently excluded from fusion because it 
//significantly underperforms the other stages and adds noise to the final score 

const NUM_FRAMES: usize = 16;

// Decision layer
#[derive(Debug)]
enum Decision { Accept, Retry, Reject }

//3 possible decisions
impl Decision {
    fn from_score(score: f64) -> Self {
        if score >= ACCEPT_THRESHOLD     { Decision::Accept }
        else if score >= RETRY_THRESHOLD { Decision::Retry  }
        else                             { Decision::Reject }
    }

    fn label(&self) -> &str {
        match self {
            Decision::Accept => "ACCEPT",
            Decision::Retry  => "RETRY",
            Decision::Reject => "REJECT",
        }
    }

    //message for user feedback
    fn message(&self) -> &str {
        match self {
            Decision::Accept => "Live user detected.",
            Decision::Retry  => "Confidence too low, please retake in better lighting.",
            Decision::Reject => "Spoof likely detected. Access denied.",
        }
    }
}

//holds model outputs
struct StageScores {
    spatial: f64,   // ResNet18 max softmax prob on face crop
    texture: f64,   //Laplacian variance
    motion:  f64,   // inter-frame differencing 
    depth:   f64,   // MiDaS depth variance
}

//run the 4 model fncts
fn run_stages(model: &PyModule, path: &str) -> PyResult<StageScores> {
    let n = NUM_FRAMES as u64;
    Ok(StageScores {
        spatial: model.call_method1("spatial_stage", (path, n))?.extract()?,
        texture: model.call_method1("texture_stage", (path, n))?.extract()?,
        motion:  model.call_method1("motion_stage",  (path, n))?.extract()?,
        depth:   model.call_method1("depth_stage",   (path, 4u64))?.extract()?,
    })
}

//combines all signals into one final number
//for image motion is 0
fn fuse(scores: &StageScores) -> f64 {
    let is_image = scores.motion == 0.0;

    if is_image {
        let active = [W_SPATIAL, W_TEXTURE, W_DEPTH]
            .iter().filter(|&&w| w > 0.0).count() as f64;
        let extra = if active > 0.0 { W_MOTION / active } else { 0.0 };
        ((W_SPATIAL + extra) * scores.spatial
            + (W_TEXTURE + extra) * scores.texture
            + (W_DEPTH   + extra) * scores.depth)
            .clamp(0.0, 1.0)
    } else {
        (W_SPATIAL * scores.spatial
            + W_TEXTURE * scores.texture
            + W_MOTION  * scores.motion
            + W_DEPTH   * scores.depth)
            .clamp(0.0, 1.0)
    }
}



fn main() -> PyResult<()> {
    let args: Vec<String> = std::env::args().collect();

    Python::with_gil(|py| {
        let sys = py.import("sys")?;
        sys.getattr("path")?.call_method1("insert", (
            0,
            "/Users/tanya/Desktop/surt/week 13-16/rust_python_inference/venv/lib/python3.12/site-packages",
        ))?;
        sys.getattr("path")?.call_method1("append", ("../python",))?;

        let model = py.import("model")?;

        if args.len() >= 2 && args[1] == "--batch" {
            let root  = if args.len() >= 3 { args[2].as_str() } else { "../data" };
            let max_n = if args.len() >= 4 { args[3].parse().unwrap_or(10) } else { 10usize };
            run_batch(model, root, max_n)?;
            return Ok(());
        }

        if args.len() < 2 {
            println!("Usage:");
            println!("  cargo run -- <path_to_image_or_video>");
            println!("  cargo run -- --batch <dataset_root> <max_n>");
            return Ok(());
        }

        run_single(model, &args[1])
    })
}

//run single file
fn run_single(model: &PyModule, path: &str) -> PyResult<()> {
    println!("multi stage liveness inference: {}\n", path);

    let s = run_stages(model, path)?;
    let score = fuse(&s);
    let decision = Decision::from_score(score);
    let is_image = s.motion == 0.0;

    println!("Stage 1 — Spatial:    {:.4}  (ResNet18 max softmax on face crop)", s.spatial);
    println!("Stage 2 — Texture:    {:.4}  (Laplacian variance on face crop)", s.texture);
    if is_image {
        println!("Stage 3 — Motion:      N/A   (single image — no temporal data)");
    } else {
        println!("Stage 3 — Motion:     {:.4}  (inter-frame differencing on face crop)", s.motion);
    }
    println!("Stage 4 — Depth:      {:.4}  (MiDaS depth variance — weight=0 stub mode)", s.depth);
    println!();

    if is_image {
        let active = [W_SPATIAL, W_TEXTURE, W_DEPTH]
            .iter().filter(|&&w| w > 0.0).count() as f64;
        let extra = if active > 0.0 { W_MOTION / active } else { 0.0 };
        println!("Fusion (image): ({:.3})×spatial + ({:.3})×texture  [motion redistributed]",
            W_SPATIAL + extra, W_TEXTURE + extra);
    } else {
        println!("Fusion (video): {:.2}×spatial + {:.2}×texture + {:.2}×motion + {:.2}×depth",
            W_SPATIAL, W_TEXTURE, W_MOTION, W_DEPTH);
    }
    println!("Fused score:    {:.4}", score);
    println!();
    println!("Decision: {}  —  {}", decision.label(), decision.message());
    println!("{}", "-".repeat(55));

    Ok(())
}

//dataset eval
fn run_batch(model: &PyModule, dataset_root: &str, max_n: usize) -> PyResult<()> {
    let samples: Vec<(String, i64, String)> = model
        .call_method1("iter_dataset", (dataset_root,))?
        .extract()?;

    println!("Found {} samples (showing up to {})\n", samples.len(), max_n);
    println!(
        "{:<22} {:<20} {:>9} {:>7} {:>7} {:>7} {:>7} {:>6} {:<12} {}",
        "File", "Folder", "Spatial", "Texture", "Motion", "Depth", "Score", "Label", "Decision", "Outcome"
    );
    println!("{}", "-".repeat(135));

    let mut true_accepts  = 0usize;
    let mut true_rejects  = 0usize;
    let mut false_accepts = 0usize;
    let mut false_rejects = 0usize;
    let mut retry_live    = 0usize;
    let mut retry_spoof   = 0usize;
    let mut total = 0usize;
    let mut category_stats: HashMap<String, (usize, usize, usize)> = HashMap::new();


    ////outcome logic
    for (path, label, folder) in samples.iter().take(max_n) {
        match run_stages(model, path) {
            Ok(s) => {
                let score    = fuse(&s);
                let decision = Decision::from_score(score);
                let is_live  = *label == 1;
                let is_image_live = is_live && s.motion == 0.0;

                let outcome = match (&decision, is_live) {
                    (Decision::Accept, true)  => { true_accepts  += 1; "✓ TA" }
                    (Decision::Accept, false) => { false_accepts += 1; "✗ FA" }
                    (Decision::Reject, true)  => { false_rejects += 1; "✗ FR" }
                    (Decision::Reject, false) => { true_rejects  += 1; "✓ TR" }
                    (Decision::Retry,  true)  => { retry_live    += 1; "~ RL" }
                    (Decision::Retry,  false) => { retry_spoof   += 1; "~ RS" }
                };
                total += 1;

                let entry = category_stats.entry(folder.clone()).or_insert((0, 0, 0));
                entry.2 += 1;
                match outcome {
                    "✓ TA" | "✓ TR" => entry.0 += 1,
                    "~ RL" | "~ RS" => entry.1 += 1,
                    _ => {}
                }

                let filename = std::path::Path::new(path)
                    .file_name().and_then(|f| f.to_str()).unwrap_or(path);

                let pl = path.to_lowercase();
                let motion_str = if s.motion == 0.0
                    && (pl.ends_with(".jpg") || pl.ends_with(".jpeg") || pl.ends_with(".png"))
                { "  N/A".to_string() } else { format!("{:.3}", s.motion) };

                let outcome_str = if is_image_live {
                    format!("{} [no motion]", outcome)
                } else {
                    outcome.to_string()
                };

                println!(
                    "{:<22} {:<20} {:>9.3} {:>7.3} {:>7} {:>7.3} {:>7.3} {:>6} {:<12} {}",
                    &filename[..filename.len().min(22)],
                    &folder[..folder.len().min(20)],
                    s.spatial, s.texture, motion_str, s.depth, score,
                    if is_live { "live" } else { "spoof" },
                    decision.label(),
                    outcome_str,
                );
            }
            Err(e) => eprintln!("  [ERROR] {} — {:?}", folder, e),
        }
    }

    println!("{}", "-".repeat(135));
    print_summary(
        total, true_accepts, true_rejects, false_accepts,
        false_rejects, retry_live, retry_spoof, &category_stats,
    );

    let _ = model.call_method0("print_crop_stats");

    Ok(())
}

fn print_summary(
    total: usize,
    true_accepts: usize, true_rejects: usize,
    false_accepts: usize, false_rejects: usize,
    retry_live: usize, retry_spoof: usize,
    category_stats: &HashMap<String, (usize, usize, usize)>,
) {
    if total == 0 { return; }

    let definitive = true_accepts + true_rejects + false_accepts + false_rejects;
    let retried    = retry_live + retry_spoof;
    let correct    = true_accepts + true_rejects;

    println!("\nPer-category breakdown:");
    println!("  {:<25} {:>8} {:>8} {:>8}", "Folder", "Correct", "Retried", "Total");
    println!("  {}", "-".repeat(52));
    let mut keys: Vec<_> = category_stats.keys().collect();
    keys.sort();
    for k in keys {
        let (c, r, t) = category_stats[k];
        let ltype = if k.to_lowercase().starts_with("live") { "live" } else { "spoof" };
        println!("  {:<25} {:>8} {:>8} {:>8}  [{}]", k, c, r, t, ltype);
    }

    println!("\nOutcome key: TA=true accept  TR=true reject  FA=false accept  FR=false reject  RL=retry(live)  RS=retry(spoof)");
    println!("             [no motion] = live_selfie static image, motion stage structurally unavailable");
    println!();
    println!("Overall ({} samples — {} definitive, {} retried):", total, definitive, retried);
    println!("  True  accepts (live  → ACCEPT): {:>3}  |  True  rejects (spoof → REJECT): {:>3}",
        true_accepts, true_rejects);
    println!("  False accepts (spoof → ACCEPT): {:>3}  |  False rejects (live  → REJECT): {:>3}  <- security / UX failures",
        false_accepts, false_rejects);
    println!("  Retry on live:                  {:>3}  |  Retry on spoof:                 {:>3}  <- attacker gets another attempt",
        retry_live, retry_spoof);
    println!();

    if definitive > 0 {
        println!("  Definitive accuracy: {}/{} ({:.1}%)", correct, definitive,
            100.0 * correct as f64 / definitive as f64);
        if false_accepts + true_rejects > 0 {
            let far = false_accepts as f64 / (false_accepts + true_rejects) as f64 * 100.0;
            println!("  FAR (spoof accepted): {:.1}%  <- lower is better", far);
        }
        if false_rejects + true_accepts > 0 {
            let frr = false_rejects as f64 / (false_rejects + true_accepts) as f64 * 100.0;
            println!("  FRR (live rejected):  {:.1}%  <- lower is better", frr);
        }
    }

    println!();
    println!("Fusion weights: spatial={} texture={} motion={} depth={}",
        W_SPATIAL, W_TEXTURE, W_MOTION, W_DEPTH);
}