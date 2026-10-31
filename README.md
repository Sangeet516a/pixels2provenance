# From Pixels to Provenance

**Automated galaxy morphology classification in archival Hubble Space Telescope images.**

NUS-ISS Graduate Certificate in Pattern Recognition Systems, Practice Module
Sangeeta Bahrani (A0339780R) · Supervisor: Zhu Fangming

The project takes calibrated HST images in FITS format and produces one catalogue, `catalog.csv`, in which every
detected source says what it is, how sure the model is, and where each value came from. It also writes a colour
PNG of every classified galaxy.

The full method and results are in the
[project report](report/Pixels2Provenance_Project_Report_31Oct_FINAL.docx). The history of every change is in
[`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).

---

## What the project does

Raw archive images mix galaxies, stars and false detections such as cosmic rays. Morphology labels exist for only
some galaxies, and they come from volunteer votes rather than fixed classes. The pipeline handles this in six steps:

1. **Prepare the images.** It checks frame quality, detects and deblends sources, aligns the filters, rejects
   cosmic rays, matches the PSF across filters and measures morphology with `statmorph`.
2. **Separate sources without labels.** A convolutional autoencoder, a Gaussian mixture model, HDBSCAN outlier
   scores and physical rules sort every source into Galaxy, Star or Noise. Each change made by a rule is recorded.
3. **Use the stars for calibration.** Clean stars build an empirical PSF for each pointing, and the galaxies are
   then measured again with it.
4. **Classify the galaxies.** Labels are derived from Galaxy Zoo: Hubble (GZH) vote fractions. Four models are
   averaged with equal weight and calibrated: fine-tuned ResNet-18, fine-tuned EfficientNet-B0, an XGBoost on 18
   physical features, and a fusion XGBoost on frozen CNN features plus those physical features. A separate XGBoost
   gives a merger probability.
5. **Test the result.** The model is scored on held-out sky regions with the same GZH-derived labels, and on the
   independent CANDELS expert catalogue (Kartaltepe et al. 2015), where agreement between the two human catalogues
   sets the ceiling.
6. **Deliver.** It writes a 95-column catalogue with calibrated probabilities and provenance flags, plus a galaxy
   image atlas sorted by predicted class.

**Classes:** Smooth, FaceOnSpiral, EdgeOn, Irregular and Ambiguous, plus a separate merger probability.

## Final results

Final run of 31 October 2026: 14 GOODS pointings, 42 FITS files, about 4 hours on an Apple-silicon Mac.

| Evaluation | Galaxies | Macro-F1 (95 % CI) | Accuracy | Galaxies given a shape |
|---|---:|---:|---:|---:|
| Out-of-fold (train + dev) | 3,185 | 0.551 (0.533–0.569) | 0.556 | 85.2 % |
| **Test** | **328** | **0.548 (0.494–0.599)** | **0.543** | **88.1 %** |
| External sky | 434 | 0.562 (0.515–0.606) | 0.576 | 85.7 % |

| Other measure | Result | Proposal tier |
|---|---|---|
| Morphology Macro-F1 (Target 0.65; majority-class baseline 0.084) | 0.548 on test | Minimum |
| Merger PR-AUC (test prevalence 0.030, only 10 test mergers) | 0.131 on test | Minimum |
| Calibration ECE | 0.074 on test | Minimum |
| Ensemble gain over the best single source | +0.023 | Minimum |
| Learned star finder vs baseline, F1 against Gaia | 0.452 vs 0.608 | Minimum not met |
| CANDELS: human-human ceiling and cross-catalogue Macro-F1 | 0.276 and 0.227 | Gap quantified |

**Catalogue:** 42,827 rows and 95 columns: 13,242 galaxies (12,453 classified), 254 stars and 29,331 noise rows.
4,254 galaxies have a GZH label, and 2,415 are matched to CANDELS.

**Limits of these results:**

- The scores measure agreement with labels derived from volunteer votes, not physical truth.
- The 8,199 galaxies with a model prediction only are fainter than the labelled ones, so they are not directly
  scored.
- The test split was evaluated four times during development, and every evaluation is recorded in
  [`test_split_ledger.json`](test_split_ledger.json). An external score was also seen once before the final run.
  No model choice used those scores.

Section 14 of the report gives the full discussion.

---

## Requirements

- Python 3.10–3.13
- About 10 GB of disk space for the 42 FITS files and another 10 GB for the outputs of a full run
- A GPU is optional: Apple MPS or CUDA is used automatically, and a CPU works but is much slower
- Internet access for downloading images and catalogues, and ImageNet weights on the first run (about 66 MB)

## Setup

```bash
git clone https://github.com/Sangeet516a-nus/pixels2provenance.git
```

```bash
cd pixels2provenance
```

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, use `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

`requirements-lock.txt` lists the exact versions used for the final run. Use it instead if you load the saved models,
which can depend on those versions (Python 3.13.9):

```bash
pip install -r requirements-lock.txt
```

If `statmorph` fails to build:

```bash
pip install --no-build-isolation statmorph
```

The FITS images and catalogues are too large for GitHub, so `data/raw/` and `data/catalogs/` are not in the
repository. Download them with the next two sections.

## Download the reference catalogues

```bash
python tools/download_catalogs.py gzh
```

```bash
python tools/download_catalogs.py candels
```

```bash
python tools/download_catalogs.py gaia --ra 53.117 --dec -27.754 --radius 0.2
```

```bash
python tools/download_catalogs.py gaia --ra 189.228 --dec 62.238 --radius 0.2
```

Then check the column mapping:

```bash
python tools/check_catalog_columns.py
```

| Catalogue | Role | Required? |
|---|---|---|
| Galaxy Zoo: Hubble (Willett et al. 2017) | Training labels and galaxy evidence | Yes, unless running in apply mode |
| Gaia DR3 | Star confirmation | Optional |
| CANDELS (Kartaltepe et al. 2015) | Expert validation only, never training | Optional |

## Download FITS images

### Any sky position

`tools/get_fits.py` searches the MAST archive and downloads one image per filter. It only downloads, and does not
run the pipeline.

```bash
python tools/get_fits.py --target <RA>,<DEC>,<FIELD_NAME> --filters <FILTER_1> <FILTER_2> <FILTER_3> --out <TARGET_FOLDER>
```

| Part | Meaning | Example |
|---|---|---|
| `<RA>` | Right ascension in degrees | `189.15` |
| `<DEC>` | Declination in degrees (keep the minus sign for southern positions) | `62.20` or `-27.80` |
| `<FIELD_NAME>` | Short name, which also becomes the sub-folder name | `MY-FIELD` |
| `<FILTER_n>` | Filters to download; leave out `--filters` to choose one per colour automatically | `F435W F606W F775W` |
| `<TARGET_FOLDER>` | Parent folder; files go to `<TARGET_FOLDER>/<FIELD_NAME>/` | `data/fields` |

Do not put spaces around the commas in `--target`.

Check what would be downloaded first. `--list` downloads nothing:

```bash
python tools/get_fits.py --target 189.15,62.20,MY-FIELD --filters F435W F606W F775W --out data/fields --list
```

Then remove `--list` to download (each file is about 200–450 MB):

```bash
python tools/get_fits.py --target 189.15,62.20,MY-FIELD --filters F435W F606W F775W --out data/fields
```

Other useful options:

| Option | Use |
|---|---|
| `--target ...` repeated | Download several positions in one command |
| `--instrument` | `ACS/WFC` (default), `WFC3/UVIS` or `WFC3/IR` |
| `--gaia` | Also save the Gaia DR3 stars around the position |
| `--allow-single` | Accept single-exposure images when nothing else exists (expect many more cosmic rays) |
| `--radius-arcmin` | Search radius, 2.0 by default |

The filters are chosen together, to maximise the sky area that all of them cover. Images made from several
exposures come first, because they already have cosmic rays removed.

### The exact 42 files of the final run

To reproduce the report, use these exact files, because the split plan in `config.yaml` matches them by name.
`get_fits.py` may choose different images for the same position. This loop downloads about 10 GB into `data/raw/`:

```bash
mkdir -p data/raw && for f in hst_11563_34_acs_wfc_f775w_jb5x34tc hst_11563_4i_acs_wfc_f606w_jb5x4ilh hst_9425_11_acs_wfc_f435w_j8e611rb hst_9583_a1_acs_wfc_f435w_j8dna1ar hst_9583_h8_acs_wfc_f606w_j8dnh8we hst_9583_h8_acs_wfc_f775w_j8dnh8wn hst_9803_33_acs_wfc_f606w_j8mt33ek hst_9803_39_acs_wfc_f435w_j8mt39ot hst_9803_41_acs_wfc_f775w_j8mt41fc j8dn99010 j8dna0010 j8dna3010 j8dna4010 j8dna6010 j8dna9010 j8dnb4010 j8dnb4030 j8dnb5010 j8dnb5030 j8dnb8010 j8dnb8030 j8dnb9010 j8dnb9030 j8dnc1010 j8dnc4010 j8dnc4030 j8dnf2030 j8e606010 j8e607010 j8e609010 j8e613010 j8e621010 j8e621030 j8e622010 j8e622030 j8e624010 j8e624030 j8e628010 j8e628030 j8e688010 j8e688030 jb5x4k010; do curl -fL -o "data/raw/${f}_drc.fits" "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/${f}_drc.fits"; done
```

| Pointing | RA, Dec (deg) | Used for |
|---|---|---|
| GOODS-S | 53.180, -27.760 | train, dev, external stripes |
| GOODS-S-2 | 53.015, -27.814 | test, dev stripes |
| GOODS-S-3, -4, -5, -8, -9 | around 53.1, -27.8 | train |
| GOODS-N-1, -6, -9 | around 189.3, 62.3 | train |
| GOODS-N-2 | 189.310, 62.277 | train, test, external stripes |
| GOODS-N-3 | 189.066, 62.163 | train, dev stripes |
| GOODS-N-4 | 189.391, 62.315 | test, external stripes |
| GOODS-N-7 | 189.065, 62.239 | external |

All pointings use ACS/WFC F435W, F606W and F775W. GOODS-S, GOODS-S-2 and GOODS-N-1 use images with few exposures,
so they contain far more cosmic rays than the others.

## Run the project

Reproduce the final run (about 4 hours on an Apple-silicon Mac, much longer on a CPU):

```bash
python run_pipeline.py
```

Stop and resume as needed:

| Command | Use |
|---|---|
| `python run_pipeline.py --list` | Show all stages and their output folders |
| `python run_pipeline.py --from 4a` | Run from a stage to the end |
| `python run_pipeline.py --from 1a --to 1e` | Run a range of stages |
| `python run_pipeline.py --only 1c` | Run one stage (its inputs must already exist) |
| `python tools/show_stage.py 2d` | Show what a stage did, its files and its numbers |
| `python tools/inspect_catalog.py` | Summarise a finished `catalog.csv` |

**Do not edit `src/` or `config.yaml` during a run.** A stage that starts later would use the changed code.

Results vary slightly between runs, because the autoencoders and networks are retrained and GPU training is not
bit-reproducible. To compare with the final run, check these log lines:

- Stage 1a: `42 verified file(s), 0 rejected`
- Stage 8: `catalog.csv: 42827 rows x 95 columns`

### Run on a new field

1. Download the images with `tools/get_fits.py`, as above.
2. Move the old files out of `data/raw/`, and copy the new field's FITS files into it.
3. In `config.yaml`, set `partition.pointing_splits: []` and `partition.external_patterns: []`, because the
   current values name the files of the final run. Stage 0 then assigns the splits automatically.
4. Run `python run_pipeline.py`.

To do these steps in one go, use the field runner. It moves the previous run to `runs/<time>/`:

```bash
python tools/run_field.py --target <RA>,<DEC>,<FIELD_NAME> --search-only
```

```bash
python tools/run_field.py --target <RA>,<DEC>,<FIELD_NAME>
```

**Train or apply.** With at least 150 GZH galaxies in the field, the models are trained on it. Otherwise a saved
model is applied, which you export from a finished run:

```bash
python tools/export_model.py --name <MODEL_NAME>
```

Apply mode and the field runner were not re-tested on the final model. Check `--search-only` before the
first real download. GZH labels exist only for AEGIS, COSMOS, GEMS and GOODS, and CANDELS is used only for
GOODS-South in this project.

## Final model

The final model is submitted as `pixels2provenance_final_2026-10-31.zip`. It is not stored in this repository, whose
`.gitignore` excludes `models/`.

| Tier | Contents |
|---|---|
| Tier 1, unsupervised | Quality and representation autoencoders, PCA and scalers, GMM, HDBSCAN, cluster names |
| Tier 2, supervised | Fine-tuned ResNet-18 and EfficientNet-B0, physical and fusion XGBoost, merger XGBoost, calibration |

To use it, unzip it into `models/`. `MODEL_README.md` inside the zip explains each file and how to apply the model to
a new field. Loaded from the package, Tier 2 reproduced all 328 final test predictions exactly, and the GMM reproduced
the cluster of all 43,015 sources.

## Outputs

Each stage writes to its own folder in `outputs/`, which is not in the repository because of its size.

| File | Content |
|---|---|
| `outputs/stage8_catalogue/catalog.csv` | **The final catalogue** |
| `outputs/stage8_catalogue/catalog_columns.csv` | Description of every column and the rows it fills |
| `outputs/stage9_galaxy_atlas/galaxy_images.zip` | **Colour PNG of every classified galaxy, by predicted class** |
| `outputs/stage6a_test_eval/test_metrics.json` | Test scores and tiers |
| `outputs/stage7b_crosscat/external_gzh_metrics.json` | External scores, overall and per pointing |
| `outputs/stage7c_generalisation/generalisation_report.json` | CANDELS ceiling and generalisation gap |
| `outputs/stage4d_ablation/ablation_table.csv` | What each model and input adds |
| `outputs/pipeline.log`, `outputs/run_log.json` | Full log, and the status and time of each stage |

Useful catalogue columns:

| Column | Meaning |
|---|---|
| `classification` | Star, Galaxy or Noise |
| `pred_class`, `p_smooth` … `p_ambiguous` | Predicted class and calibrated probabilities |
| `is_pseudo_label` | False if the galaxy has a GZH label, True if it has a model prediction only |
| `gz_label`, `k15_class` | The GZH-derived label and the CANDELS expert class |
| `p_merger` | Merger probability, best read as a ranking |
| `classification_override`, `cr_suspect` | Why a rule changed the class, and whether the object may be a cosmic ray |

`split_tag` shows the split. Train and dev galaxies carry their out-of-fold prediction, so their catalogue
predictions never come from a model that saw them.

## Pipeline stages

There are 28 stages in ten phases. Phase 5 runs before Phase 4, so the models train on the refined measurements.

| Phase | Stages | What happens |
|---|---|---|
| 0 Governance | 0 | Split the sky into regions, with collars so that splits share no pixels |
| 1 Preprocessing | 1a–1e | Read FITS files, frame QC, detection and cosmic rays, PSF matching, first morphology |
| 2 Segregation | 2a–2e | Autoencoder, clustering, baseline, naming clusters by evidence, physical gate |
| 3 Routing | 3a–3c | GZH labels for galaxies, stars for calibration, noise model |
| 5 Calibration | 5a–5b | Empirical PSF, second morphology pass |
| 4 Supervised | 4a–4d | Galaxy stamps and CNN features, ensemble, calibration, ablation |
| 6 Evaluation | 6a–6c | Test score, uncertainty audit, SHAP and Grad-CAM |
| 7 Robustness | 7a–7c | Taxonomy ceiling, external and CANDELS scores, generalisation gap |
| 8 Catalogue | 8 | `catalog.csv`, column dictionary, delivery checks |
| 9 Atlas | 9 | Galaxy image atlas |

## Repository layout

```text
pixels2provenance/
├── run_pipeline.py          Runs the stages
├── config.yaml              All settings, including the split plan and label rules
├── requirements.txt         Python packages
├── requirements-lock.txt    Exact package versions of the final run
├── test_split_ledger.json   Every evaluation of the test split
├── DEVELOPMENT_LOG.md       History of changes and their effect
├── report/                  Project report
├── src/p2p/                 Shared code: detection, morphology, networks, inference
│   └── stages/              One file per stage, each starting with Purpose / Method / Output
├── tools/                   Download, inspection and model-export tools
└── data/
    ├── raw/                 FITS images (not in the repository)
    └── catalogs/            GZH, Gaia and CANDELS tables (not in the repository)
```

## Troubleshooting

| Message or problem | Fix |
|---|---|
| `... not found - run stage X first` | Run the earlier stage, or use `--from` |
| `No labelled galaxies in split(s) [...]` | Lower `partition.collar_arcmin`, or use apply mode if the field has too few GZH labels |
| `No Galaxy Zoo: Hubble galaxies cover this field` | Use apply mode, or choose a field GZH covers |
| `get_fits.py`: `-> nothing usable for ...` | The line above says why; try another position, `--instrument` or `--allow-single` |
| `catalogs: required column(s) not found` | Run `python tools/check_catalog_columns.py`, then fix `catalogs.<kind>.columns` |
| `BIC still falling at k_max` | Expected on this data: the knee of the BIC curve is used |
| Stage 1e or 5b is very slow | `statmorph` is slow per source; set `project.n_jobs: -1` |
| Out of memory in Stage 1c | Close other programs, or process fewer pointings at a time |

## Main references

- Willett et al. 2017, MNRAS 464, 4176: Galaxy Zoo: Hubble
- Kartaltepe et al. 2015, ApJS 221, 11: CANDELS visual classifications
- Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1: Gaia DR3
- Rodriguez-Gomez et al. 2019, MNRAS 483, 4140: statmorph
- He et al. 2016 (ResNet); Tan & Le 2019 (EfficientNet); Chen & Guestrin 2016 (XGBoost)

The report has the full list. Anthropic's Claude was used as an AI assistant during the project, as the report
states.
