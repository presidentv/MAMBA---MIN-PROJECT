# Mini Project: explainable multimodal engagement detection
 
Last updated 20 August 2026.
 
## What the project is
 
Adhvaith's academic mini project. Stated title:
 
> Explainable Multimodal AI Framework for Early Detection of Learning Engagement and Emotional States in Children Using Eye Movement Pattern Recognition and Sentiment Analysis
 
Four intended components: eye movement pattern recognition in children, emotional state recognition through sentiment analysis, Transformer-based multimodal fusion, and explainable AI over the predictions.
 
Institution is VIT (School of Computer Science and Engineering, SCOPE). Deliverables follow the VIT Review template format.
 
## Constraints (confirmed by Adhvaith)
 
- One semester, roughly twelve weeks.
- No access to child participants. No ethics approval route within the timeline.
- Working alone or in a small team. Team member names, registration numbers, course code and faculty guide name are still unknown.
Because of these constraints the agreed working scope is the reduced version: train on public adult corpora, treat age transfer as a measured result rather than an assumption, and do not claim a child data collection.
 
## Agreed technical plan
 
Data: DAiSEE as the primary corpus (9,068 clips, 112 subjects, four ordinal engagement levels, severely imbalanced). EngageNet as a second corpus if access arrives in time. EmoReact for a child affect transfer analysis.
 
Pipeline:
1. OpenFace 2.0 at 5 fps for gaze vectors, head pose, eye landmarks, action units.
2. Gaze features: fixation proxies from angular velocity thresholding, gaze dispersion, blink rate from AU45, off-screen ratio, head pose stability.
3. Affect branch: action unit intensities or frozen pretrained FER embeddings (HSEmotion was suggested).
4. Fusion: two temporal transformer encoders, two layers, d=128, cross-attention block, ordinal head over four levels.
5. Explanation: SHAP over modality-level features, attention rollout, and a deletion test for faithfulness.
Required baselines: majority class, logistic regression on mean features, each modality alone, naive late fusion. Headline metric is macro-F1 with the confusion matrix, never bare accuracy, because majority-class prediction on DAiSEE already reaches roughly 50 per cent.
 
Early detection is implemented as a label shift: predict the engagement label of clip t+1 from clip t within the same session, and report accuracy against lead time. This was identified as the highest value per unit of effort in the whole project, since almost nobody reports a prediction horizon.
 
## Survey findings
 
A structured search across Google Scholar, arXiv, Semantic Scholar, ACM DL and IEEE Xplore, covering 2021 to 2026 with seminal earlier work retained, produced these conclusions.
 
No published system combines all four target components. The closest are three-way overlaps:
- Transformer plus Bi-LSTM engagement fusion (Springer 2025): gaze, fusion, engagement, no XAI. This is the primary architectural baseline.
- Behavior capture explainable engagement recognition (Springer 2024): engagement with intrinsic XAI, no gaze or text branch.
- Interpretable multimodal emotion recognition with SHAP (J. Supercomputing 2025): fusion plus XAI on emotion, not engagement, adults only.
Every major engagement benchmark from 2021 to 2026 uses adults or university students: DAiSEE, EngageNet (ages 18 to 37), CMOSE, DIPSER, OUC-CGE. There is no child engagement corpus with the required modalities. This is simultaneously the novelty claim and the main practical obstacle.
 
Gaze and head pose are the strongest individual predictors of engagement in EngageNet, which justifies treating eye movement as the primary channel.
 
Webcam eye tracking with young children is feasible but lossy. Steffan et al. (Infancy 2024, N=125 toddlers) report about 42 per cent attrition against about 10 per cent for laboratory trackers, and accuracy only at area-of-interest resolution.
 
Zero-shot vision language models tested on classroom engagement (2026 benchmark) perform near randomly per student, collapse predictions onto one level, and swing by more than thirty accuracy points with prompt wording. This is the answer to "why not just prompt a VLM".
 
## Known weaknesses in the original framing
 
The sentiment analysis component is under-specified. Children in a learning task produce little text, child ASR is weak, and child speech emotion models barely exist. Either the text source needs defining concretely or the component should be renamed to facial affect. This was raised and is still unresolved in the project title, though the deliverables now describe the affect branch rather than a text branch.
 
A from-scratch transformer will overfit at this sample size. Frozen encoders with a light fusion head, plus a logistic regression reported beside every result, is the agreed mitigation.
 
## Regulatory and ethical position
 
Article 5 of the EU AI Act prohibits emotion inference in educational settings, and the European Commission's 2025 guidelines on prohibited practices reason explicitly about children's vulnerability and school power imbalance. The project uses existing consented corpora and collects no new child data, so it falls outside the prohibition, but the framing must position the work as research into interpretable engagement modelling rather than a deployable classroom monitoring product. European reviewers are likely to raise this.
 
Fairness exposure: common affect corpora contain under two per cent of samples in the darkest skin tone group (TrustSkin, 2025). Disaggregated evaluation should be planned from the start.
 
## Deliverables produced so far
 
All in the outputs folder.
 
- `explainable-multimodal-child-engagement/outline.yaml` and `fields.yaml`: research outline of 33 items across 8 categories, with a 37-field extraction schema. Built with the deep-research skills. The deep extraction phase has NOT been run, so `results/` is empty.
- `Related_Work_Landscape_and_Gap_Analysis.docx`: 8-page landscape and gap analysis with a coverage matrix.
- `Review1_filled_v3.pptx`: the VIT Review-1 deck, filled. Eight slides after duplicating the Literature Review slide. Contains 15 papers across two lit review tables and 16 references.
- `Proposal_IEEE.docx`: three-page IEEE two-column conference-format proposal with survey, gaps, method, novelty, evaluation, plan, ethics, references. This superseded an earlier `Project_Proposal.docx` that Adhvaith judged to look AI-generated.
## Open items
 
1. Slide 1 of the review deck still has placeholders: programme (B.Tech or M.Tech), course code and title, both team member names with registration numbers, and the faculty guide name. Same for the IEEE proposal masthead.
2. Author lists for several 2024 to 2026 references are unverified. Specifically EngageNet [5] and DIPSER [9] author lists, and entries [11] to [15] and [18] which are listed title-first because authors could not be confirmed from a single search pass. LIRIS-CSE also needs citation verification.
3. Deep research phase not run. Running it would replace descriptive summaries with verified metrics, participant counts and dataset properties.
4. Dataset access requests for DAiSEE and EngageNet have not been started. These were flagged as the single biggest schedule risk and should be submitted first.
## Working preferences observed
 
- Wants concise, direct output without padding.
- Dislikes AI-looking formatting: shaded callout boxes, colored headings, everything laid out as tables, bulleted lists everywhere. Prefers prose and standard academic layout.
- Asked for the humanizer skill to be applied to written content.
- Prefers deliverables in Word and PowerPoint, matching institutional templates exactly without altering the template design.
## Related work: MRG-VM (competing/adjacent approach)

Added 20 August 2026, from a document Adhvaith supplied titled "Motion Reliability Guided Vision Mamba (MRG-VM): A Behavioral Feature Learning Framework for Student Engagement Detection." This is a project plan, not a published paper, but it targets the same DAiSEE-style engagement problem with an adjacent architecture and should be tracked as related/competing work.

Seven-phase pipeline. Two phases are named as its novel contributions:
- Phase 1 (novel): computes a per-frame Motion Reliability Score (MRS) from blur, face visibility, head rotation, eye visibility, and motion consistency, then discards low-quality frames before any feature extraction. A quality-gating step upstream of the model, not a fusion or XAI contribution.
- Phase 2: MediaPipe Face Mesh landmark extraction (face, eye, iris, head pose, tracking).
- Phase 3 (novel): feeds only the MRS-filtered reliable frames into Vision Mamba (a state-space-model alternative to a Transformer) to learn spatial-temporal behavioral representations, producing embeddings tied to eye gaze, blink patterns, head movement, and facial dynamics.
- Phase 4: fuses Mamba embeddings with landmark-based geometric features into one normalized vector.
- Phase 5: lightweight MLP classifier on the fused vector predicts engagement level.
- Phase 6: ablation study isolating the contribution of MRS, Vision Mamba, landmark features, and fusion individually.
- Phase 7: SHAP for feature-importance visualization.

Differentiation from this project's plan: MRG-VM has no early-detection/label-shift component, no attention-rollout or deletion-test faithfulness check layered on top of SHAP (single XAI method only), and no disaggregated fairness evaluation or child-transfer analysis. Its novelty rests on frame-reliability filtering (MRS) plus a Mamba backbone instead of a Transformer, not on prediction-horizon framing or faithfulness-audited explanations. This project's existing differentiators (label-shift early detection, multi-method faithfulness-audited XAI, measured EmoReact child-transfer, VLM zero-shot baseline) remain distinct from MRG-VM.

Action items noted: cite MRG-VM as related work in the lit review / gap analysis; consider MRS-style reliability filtering as an optional, cheap, orthogonal preprocessing addition to this project's own OpenFace pipeline (it doesn't conflict with the novelty claims already agreed).

## Tooling notes
 
The deep-research skills from github.com/Weizhena/deep-research-skills are installed as five skills: research, research-add-items, research-add-fields, research-deep, research-report. They were adapted for this environment: the repo's separate web-search-agent is inlined into the subagent prompt, the validate_json.py dependency is replaced by an inline coverage check, and academic source routing (Scholar, arXiv, Semantic Scholar, ACM DL, IEEE Xplore) was added to the research and research-deep skills.