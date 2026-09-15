# Data resources for learning and validating event structure

Status: resource survey\
Checked: 2026-08-10\
Related note: [Event structure from multimodal stimulus features](event-structure-notes.md)

## Bottom line

There is no single large, clean corpus of naturalistic movies or spoken/written stories with many independent human judgments of fine and coarse narrative events. The available resources divide into five useful classes:

1. **Direct event-boundary judgments**, which are closest to the scientific target but usually small or domain-limited.
2. **Temporally localized actions, scenes, and captions**, which are much larger and useful for pretraining segmentation, duration, and event-content models, but do not measure cognitive event perception directly.
3. **Memory, attention, and importance judgments**, which provide consequential or salience-based validation without defining event boundaries.
4. **Narrative turning points and event relations**, which can supervise coarse structure, recurrence, goals, causality, and hierarchy.
5. **Large text and audiobook reservoirs**, which have editorial structure or aligned audio/text but no human event gold labels; these are suitable for self-supervision or carefully identified AI pseudo-labels.

The small direct datasets should generally be protected as validation or final-test data. Large action and scene corpora can train a boundary proposer or duration model, but their labels should not silently become the operational definition of a perceived event.

“Freely available” below includes direct public downloads, no-cost downloads after a data-use agreement, and annotation-only releases whose underlying commercial or YouTube media must be obtained separately. These cases are distinguished in the access notes.

## Best starting set

For an initial benchmark, the highest-value combination is:

- **Direct human/AI boundary targets:** Kinetics-GEBD, Kinetics-GEB+, the Michelmann narrative data, ScriptPriming, EventRecall, and the Dryad infant/adult movie data.
- **Large temporal pretraining:** ActivityNet Captions, EPIC-KITCHENS-100, Breakfast Actions, YouCook2, and Charades.
- **Movie/narrative structure:** TRECVID Deep Video Understanding, MovieNet, MovieGraphs, StudyForrest, and TRIPOD.
- **Memory and salience validation:** the Naturalistic Free Recall Dataset, TVSum, SumMe, and Sherlock recall data.
- **Causal and relational supervision:** MAVEN-ERE, CaTeRS, and GLUCOSE.
- **Spoken/written narrative processing:** Natural Stories, Le Petit Prince, Alice, GECO, BookSum, NarrativeQA, SQuALITY/QuALITY, LitBank, and TellMeWhy.
- **Large open stimulus reservoirs:** Project Gutenberg plus LibriSpeech or Multilingual LibriSpeech; Libri-Light is useful when transcripts or event labels are not required.

If stimulus redistribution and reproducibility matter, prioritize TRECVID's Creative Commons movies, Breakfast Actions, the Naturalistic Free Recall Dataset, and the research stimuli linked by the Dynamic Cognition Laboratory. YouTube-derived and commercial-film datasets are potentially larger, but media availability can decay and annotations may be the only redistributable component.

### Implications for `word2psy` and `aud2psy`

Spoken and written narratives introduce two distinct notions of sequence:

- **Presentation position or time:** token, sentence, paragraph, or chapter position for text; seconds and word timestamps for speech.
- **Narrated or story-world time:** the order and duration of the events being described.

The two diverge under flashbacks, foreshadowing, summaries, repeated descriptions, and variable narration rate. Initial models should infer boundaries along presentation time, because that is what the available features and most human keypress/reading measures index. Story-world temporal order should be represented as a separate event relation rather than used as the primary time axis.

For paired audio/text, the preferred representation is a common interval table with word or sentence spans mapped to audio onset and offset. `word2psy` can provide semantic, character, emotion, and discourse trajectories; `aud2psy` can provide prosodic, acoustic, speaker, and pause trajectories over the same intervals. The text and audio evidence should remain separate through normalization and be fused late, so that a pause or intonational phrase is evidence about an event boundary rather than its definition.

## 1. Direct boundary and segmentation judgments

| Resource | Rater signal and scale | Best use | Access and caveats |
|---|---|---|---|
| [Kinetics-GEBD](https://github.com/StanLei52/GEBD) | Taxonomy-free boundary timestamps from nearly five human annotators per video, plus agreement/consistency information. The raw annotations retain per-rater timestamps. | The largest practical source for training a generic boundary detector and for modeling annotator disagreement. | Annotations/code are public under CC BY-NC 4.0. Source clips come from Kinetics/YouTube, so some videos have disappeared. The associated [ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shou_Generic_Event_Boundary_Detection_A_Benchmark_for_Event_Segmentation_ICCV_2021_paper.pdf) is important for interpreting the construct. |
| [Kinetics-GEB+](https://yuxuan-w.github.io/GEB-plus/) | 170,000 annotated boundaries with descriptions of the status change at each boundary in 12,000 videos; a recommended filtered set contains about 40,000 boundaries. | Jointly learning *where* a boundary is and *what changed*; especially useful for explainable boundary proposals. | The project distributes annotations and extracted video frames. It explicitly recommends using these rather than relying on the original full videos. |
| [Dynamic Cognition Laboratory stimuli/data catalog](https://dcl.wustl.edu/items/type/publications/stimuli-data-available/) | A catalog of stimuli, coding files, and repositories from event-segmentation studies, including everyday activities, narrative film, continuity editing, memory, and fine/coarse segmentation. | Finding classic human event-segmentation materials, hierarchical judgments, and stimulus codes that are otherwise scattered across papers. | This is an index rather than one standardized dataset. Some entries provide stimuli and coding but not every participant-level response; rights and file contents must be checked study by study. |
| [Neural event segmentation of continuous experience in human infants](https://datadryad.org/dataset/doi%3A10.5061/dryad.vhhmgqnx1) | Individual keypress boundaries from 22 adult behavioral raters, a thresholded consensus, and adult/infant neural data for two short audiovisual/cartoon stimuli. | A compact cognitive gold set for consensus, response-time tolerance, and comparison with neural state transitions. | Public Dryad download. The record explains the stimulus sources; the audiovisual works have their own rights. |
| [ScriptPriming](https://github.com/dpmlab/ScriptPriming) | Individual human boundaries for narratives whose overlapping event scripts are manipulated by top-down attention; story text, response files, code, and links to audio are included. | Testing whether a model changes segmentation with task or prior context rather than treating boundaries as stimulus-invariant. | Public repository under CC0; audio and imaging data are linked separately. The accompanying [Current Biology paper](https://www.sciencedirect.com/science/article/pii/S0960982224012247) describes the manipulation. |
| [GPT narrative event segmentation data](https://doi.org/10.5281/zenodo.10055827) | Human event-boundary judgments for three spoken stories, together with GPT-3 segmentations and a continuous model-derived boundary signal. One constituent study collected boundary judgments from 205 listeners for *Pieman*. | The clearest paired human/AI rater corpus for calibrating AI pseudo-labels against human agreement on naturalistic narratives. | Public Zenodo data; analysis code is on [GitHub](https://github.com/s-michelmann/GPT_event_segmentation). The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11810054/) documents prompts, model measures, and samples. |
| [EventRecall](https://github.com/ryanapanela/EventRecall) | Human, GPT, and LLaMA segmentations of narratives, plus human recalls and recall scores. | Comparing segmentations across human and AI raters and asking whether event definitions predict later recall. | Public repository with data and scripts; code is MIT-licensed. Generated/processed files can differ slightly because of model nondeterminism and manual formatting. See the [paper](https://doi.org/10.1038/s44271-025-00359-7). |

The Red Balloon study is also unusually relevant conceptually: 24 participants supplied fine and coarse boundaries for an extended narrative film, while the film was independently coded for changes in space, objects, characters, interactions, causes, goals, and cuts. The [open article](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2010.00168/full) fully describes those variables; the Dynamic Cognition Laboratory catalog is the best place to check which associated stimuli and coding files remain downloadable.

## 2. Large temporally localized video annotations

These are strong sources for pretraining event proposals, event descriptions, duration priors, recurring activity types, or hierarchical segmentation. They are not substitutes for human cognitive-boundary validation.

| Resource | Supervision and scale | Best use | Access and caveats |
|---|---|---|---|
| [TRECVID past datasets: Deep Video Understanding](https://trecvid.nist.gov/past.data.table.html) | The 2020–2023 development data include 14 Creative Commons movies totaling about 17.5 hours, with movie- and scene-level annotations of entities, relationships, interactions, and sentiments. | Probably the best legally tractable bridge from action datasets to actual long-form narrative movies. | NIST distributes data under the terms shown on the page; some later KinoLorber movies require a data agreement and are not Creative Commons. |
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2025) | Roughly 100 hours of unscripted egocentric cooking, around 90,000 action segments and 20,000 narrations, with verb/noun classes. | Training event-instance boundaries, recurring event types, duration models, and multimodal alignment in continuous activity. | Free for research under project terms; annotations and video downloads are provided. Procedural micro-actions are not narrative events. |
| [Breakfast Actions](https://serre.lab.brown.edu/breakfast-actions-dataset.html) | About 77 hours of everyday cooking with fine and coarse temporal segmentation, 48 action units, 52 people, and 18 kitchens. | Particularly useful for testing explicit hierarchy and HSMM duration/state recurrence. | Public under CC BY 4.0. It is procedurally narrow but unusually clean for reproducible feature extraction. |
| [YouCook2](https://youcook2.eecs.umich.edu/) | 2,000 long untrimmed cooking videos, 176 hours, 89 recipes, and temporally localized procedure steps paired with imperative descriptions. | Learning event content, temporal localization, and longer procedural dependencies. | Annotations and download instructions are public. Videos are YouTube-derived, so link rot and source rights remain issues. |
| [Charades](https://prior.allenai.org/projects/charades) | 9,848 home-activity videos, 66,500 temporal action annotations over 157 classes, 27,847 descriptions, and multi-worker consensus. | Factorial or overlapping action tracks, multilabel event content, and human-description alignment. | Public project download under its terms. Activities were enacted from prompts, so the distribution is less natural than documentary or film data. |
| [ActivityNet Captions](https://arxiv.org/abs/1705.00754) | About 20,000 long YouTube videos, 849 hours, and 100,000 human descriptions, each with a start and end time; segments can overlap. | Large-scale event proposal plus language grounding, with useful variation in event duration. | Annotations are widely available, but raw videos inherit YouTube availability. [ActivityNet-Entities](https://github.com/facebookresearch/ActivityNet-Entities) adds 158,000 boxes grounding noun phrases within the described events. |
| [Ego4D](https://ego4d-data.org/) | More than 3,700 hours of first-person video with narrations and temporal tasks such as Moments Queries and Natural Language Queries. | Scaling recurring activity/event types, long-context localization, and query-conditioned event definitions. | No-cost but gated by a data-use license/account. Large and ecologically varied, but first-person daily activity differs from edited narrative. |
| [MovieNet](https://movienet.github.io/) | 1,100 movies with 42,000 scene boundaries, 1.1 million character boxes, 65,000 place/action tags, cinematic-style labels, and movie-segment/synopsis alignments. | Long-form scene structure, character/location continuity, editorial cues, and coarse synopsis alignment. | Annotations and pretrained features are downloadable; the commercial movies themselves are not redistributed. Scene boundaries are editorial units, not perceived events. |
| [MovieGraphs](https://moviegraphs.cs.toronto.edu/) | Graph annotations for 7,637 movie clips covering characters, attributes, relationships, interactions, topics, motivations, and reasons, with many elements grounded in time. | Training interpretable situation-model tracks and causal/intentional event relations. | Data access is provided by the project; underlying film clips and derived media require attention to research-use terms and copyright. |
| [StudyForrest](https://www.studyforrest.org/data.html) | Dense annotation of *Forrest Gump*: 870 shots/cuts; locations at several abstraction levels; audio-description text and word/phoneme timing; portrayed emotions; lies, irony, sarcasm; body contact; music; eye tracking; and neuroimaging. | A rich single-film testbed for late fusion and explainable boundary evidence across many feature families. | Public research dataset, but the feature film has separate rights. Its strength is annotation depth rather than number of stimuli. |
| [AVA](https://research.google.com/ava/) | 430 fifteen-minute movie clips with 1.62 million atomic action labels across 80 classes; related releases include active-speaker and speech segments. | Character/action/speech continuity and overlapping local activity labels in edited film. | Annotations are public; source clips are movie-derived YouTube content. Atomic actions are a weak event analogue. |

## 3. Human memory, attention, and salience analogues

These resources can validate whether proposed events matter to people, even when they do not provide explicit boundary labels.

| Resource | Human signal | Best use | Access and caveats |
|---|---|---|---|
| [Naturalistic Free Recall Dataset](https://www.nature.com/articles/s41597-024-04082-6) / [OSF data](https://osf.io/h2pkv/) | 229 participants heard four 8–13 minute spoken narratives; the release includes stimulus audio/transcripts and high-fidelity, human-reviewed, time-stamped recall transcripts. | Event salience, omission, order, compression, and the relation between proposed boundaries and memory. | Data are released under CC0. This is one of the cleanest complete narrative packages, but recall units must be aligned to stimulus events. |
| [Sherlock open movie/recall data](https://openneuro.org/datasets/ds001132/versions/1.0.0) | Movie-viewing neuroimaging plus spoken free-recall data; later work released word-timestamped transcripts and alignments. | Testing event reinstatement, shared recall structure, and memory consequences of model segments. | Public OpenNeuro data; the underlying television episode remains copyrighted. See the [transcript/alignment article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10460947/). |
| [Narratives](https://openneuro.org/datasets/ds002345) | 345 participants, 891 scans, 27 spoken stories, about 4.6 hours of unique audio, and time-stamped phoneme/word transcripts. | Neural state-transition validation and transfer across many spoken narratives; some stories overlap with independently segmented corpora. | Public OpenNeuro dataset with stimuli and derivatives; most stories do not themselves have explicit boundary ratings. See the [data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC8479122/). |
| [TVSum](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) | 50 videos, each rated by 20 people for the importance of successive two-second segments. | A graded human salience curve that can be compared with event boundaries, surprise, and within-event peaks. | The original Yahoo distribution is no longer the most stable access path; public mirrors exist, including a combined [SumMe/TVSum Zenodo release](https://zenodo.org/records/4884870). Videos are largely YouTube-derived. |
| [SumMe](https://cove.thecvf.com/datasets/615) | 25 videos with at least 15 human-created video summaries each, 390 summaries in total. | Validating whether proposed event units capture material people retain in summaries. | Public research dataset and included in the [Zenodo mirror](https://zenodo.org/records/4884870). Small, but multiple judgments per stimulus are valuable. |
| [DIEM](https://thediemproject.wordpress.com/videos-and%C2%A0data/) | Frame-level eye movements from a large volunteer pool over dynamic videos. | Attention shifts, boundary-linked orienting, and detecting cases where visual salience diverges from semantic event change. | Videos and eye-movement data are available from the project. Gaze is an indirect signal and should not be treated as segmentation ground truth. |

## 4. Coarse narrative structure, causal relations, and event schemas

| Resource | Annotation signal and scale | Best use | Access and caveats |
|---|---|---|---|
| [TRIPOD](https://github.com/ppapalampidi/TRIPOD) / [Edinburgh archive](https://datashare.ed.ac.uk/handle/10283/3820) | Human turning-point annotations for 99 movies, screenplay scenes, plot synopses, and released multimodal features for an expanded movie set. | Coarse narrative boundaries, turning points, hierarchy, and screenplay-to-summary alignment. | Public research download; screenplays, plots, and commercial movies carry source-specific rights. Turning points are intentionally much coarser than ordinary events. |
| [SummScreen](https://www.tensorflow.org/datasets/catalog/summscreen) | Roughly 22,000 television transcripts paired with human-written episode recaps. | Weak supervision for which events are narratively important and how detailed sequences compress into summaries. | Freely packaged through TensorFlow Datasets. Recaps are not timestamped boundaries and source-text rights should be reviewed before redistribution or model release. |
| [MovieSum](https://aclanthology.org/2024.findings-acl.239/) | About 2,200 manually formatted movie screenplays paired with Wikipedia plot summaries and identifiers. | Large screenplay-level salience and coarse event-structure supervision. | The paper and linked resources are public; check the repository's current terms and the rights of individual screenplay sources before bulk training or redistribution. |
| [MAVEN-ERE](https://github.com/THU-KEG/MAVEN-ERE) | More than 100,000 event-coreference chains, 1.2 million temporal relations, about 58,000 causal relations, and about 16,000 subevent relations in documents. | Training event identity, recurrence/coreference, temporal order, causality, and hierarchy heads independently of perceptual features. | Public annotations and code. The domain is news rather than narrative film, so it supplies relation structure, not boundary validation. See the [EMNLP paper](https://aclanthology.org/2022.emnlp-main.60/). |
| [CaTeRS](https://cs.rochester.edu/nlp/rocstories/CaTeRS/) | Human temporal and causal relations among events in 320 short ROCStories. | A small, clean narrative test set for event-order and causal-relation proposals. | Public Brat annotations. It is too small for modern representation pretraining but useful for targeted evaluation. |
| [GLUCOSE](https://aclanthology.org/2020.emnlp-main.370/) | Roughly 670,000 crowdsourced story-specific causal explanations and generalized rules across dimensions such as motivations, emotions, and consequences. | Learning priors over goals, causes, effects, and character-state changes that can enrich event models. | Publicly downloadable through linked dataset hosts. Verify the current host's license metadata before redistributing a derived training bundle. |
| [ATOMIC 2020](https://github.com/allenai/comet-atomic-2020) | A large event-centered commonsense graph containing social, physical, and event relations. | Pretraining causal, intentional, and likely-next-event priors for SEM-like or relation-scoring models. | Public repository and models. The statements are decontextualized commonsense, not ratings tied to a particular stimulus or timepoint. |
| [STORIUM](https://storium.cs.umass.edu/) | 6,000 long collaboratively authored stories totaling about 125 million tokens, with character, goal, challenge, and other annotation cards interspersed through the narratives. | Learning long-range goal/state tracks and schema-conditioned transitions. | Free for research after accepting a data-transfer/use agreement; redistribution to people outside the research team is prohibited. |

## 5. Spoken and written narrative resources

Several of the strongest narrative resources already appear above: ScriptPriming, the Michelmann GPT/human segmentations, EventRecall, the Naturalistic Free Recall Dataset, Narratives, TRIPOD, SummScreen, MovieSum, CaTeRS, GLUCOSE, and STORIUM. This section adds resources especially useful for `aud2psy`, `word2psy`, or paired audio/text analysis.

### 5.1 Narratives with moment-by-moment human or neural responses

These provide indirect processing signals at word or audio timescales. Reading time, gaze, or shared neural response is not an event-boundary judgment, but each can test whether proposed boundaries coincide with processing difficulty, updating, attention, or cross-participant state change.

| Resource | Human signal and scale | Best use | Access and caveats |
|---|---|---|---|
| [Natural Stories](https://github.com/languageMIT/naturalstories) | Ten roughly 1,000-word stories, word-by-word self-paced reading times from about 181 readers, detailed linguistic annotations, and aligned recordings by two speakers. | An unusually useful paired `word2psy`/`aud2psy` testbed: the same token sequence can be analyzed as text, speech, reading-time response, and linguistic structure. | Public repository. Use the current release: a May 2025 update corrected a one-position misalignment in the original self-paced-reading data, in addition to an earlier Story 3 correction. These are psycholinguistically engineered stories rather than unconstrained literary prose. |
| [Le Petit Prince multilingual fMRI corpus](https://openneuro.org/datasets/ds003643) | 112 listeners—49 English, 35 Chinese, and 28 French—heard native-language versions of the same audiobook; the release includes audio, time-aligned speech/word annotations, prosodic information, comprehension quizzes, and imaging. | Cross-language validation of semantic and acoustic event evidence, narrator-paced presentation time, and neural state transitions on the same story. | Public OpenNeuro dataset. The three translations are comparable narratives, not token-identical stimuli, so cross-language alignment should be done at sentence or event level. See the [data descriptor](https://www.nature.com/articles/s41597-022-01625-7). |
| [Alice audiobook fMRI data](https://openneuro.org/datasets/ds002322/versions/1.0.3) | Raw and preprocessed fMRI from 29 people listening to the first chapter of *Alice's Adventures in Wonderland*, with code and audiobook stimulus materials; related Alice releases include EEG. | A public-domain literary narrative for testing word-aligned semantic/acoustic predictors against continuous neural response. | Public OpenNeuro dataset. It is one chapter and does not include direct event judgments. The [lab data page](https://sites.lsa.umich.edu/cnllab/2020/05/18/data-sharing-fmri-whole-brain-datasets-from-alice-in-wonderland/) links the source and paper. |
| [GECO](https://expsy.ugent.be/downloads/geco/) | Word-level eye tracking from 14 English monolinguals and 19 Dutch-English bilinguals reading the complete Agatha Christie novel *The Mysterious Affair at Styles* in English and/or Dutch. | Long-range processing validation across a full novel; testing whether inferred event updates predict rereading, fixation, and reading-time changes beyond lexical difficulty. | The project distributes English/Dutch materials, monolingual/L1/L2 reading data, and participant information. Eye movements are indirect and the bilingual design introduces language-order effects that should be modeled. |

For spoken narratives, the Michelmann/Zenodo, EventRecall, ScriptPriming, Naturalistic Free Recall, and Narratives datasets remain the first choices when direct segmentation, recall, or multiple listeners are required. Natural Stories, Le Petit Prince, and Alice add high-resolution presentation-time signals but should not be promoted to boundary gold labels.

### 5.2 Written narratives with human or AI structural supervision

| Resource | Supervision and scale | Best use | Access and caveats |
|---|---|---|---|
| [BookSum](https://github.com/salesforce/booksum) | Human-written summaries aligned at paragraph, chapter, and book levels across novels, plays, and stories; the published collection reports about 143,000 paragraph, 12,000 chapter, and 436 book examples. | Multiscale event salience and hierarchy: paragraph summaries approximate local event content, chapter summaries intermediate structure, and book summaries global plot structure. | The official repository was archived in June 2026 but remains readable and includes code, alignments, and a public bucket of chapterized Gutenberg books. Its BSD license covers code; the repository's legal note requires research use and source-specific rights compliance for collected summaries. |
| [NarrativeQA](https://github.com/google-deepmind/narrativeqa) | 1,572 books and movie scripts with Wikipedia summaries and 46,765 human questions with two answers each. | Testing whether proposed events preserve long-range causal, temporal, character, and goal information needed for narrative comprehension. Summaries can provide weak event-salience targets. | Dataset files are Apache 2.0, but full stories are retrieved from source URLs and retain their own terms. The release includes document metadata, summaries, QA pairs, and a story-download script. |
| [NarraSum](https://github.com/zhaochaocs/narrasum) | About 122,000 narrative documents drawn from movie and television plot descriptions, paired with abstractive summaries. | Large weak supervision for salient events, characters, causal structure, and summary-to-source alignment without requiring raw audiovisual stimuli. | Public repository and dataset link. It is derived from plot descriptions rather than primary stories, so events are already compressed and may reflect recap conventions. Check the current data-host license before redistribution. |
| [SQuALITY](https://github.com/nyu-mll/SQuALITY) | 127 public-domain short stories of roughly 4,000–6,000 words. Each has five questions—including “What is the plot?”—and four independently written reference summaries per question, plus human evaluation data. | High-quality multi-reference targets for global plot content, question-conditioned event salience, and agreement/disagreement among human summaries. | Stories retain their Project Gutenberg terms; summaries are CC BY. Use the corrected v1.3 release. The collection is modest but unusually well controlled. |
| [QuALITY](https://github.com/nyu-mll/quality) | Long public-domain stories averaging about 5,000 tokens with human-written multiple-choice questions, majority-vote answers, per-rater responses, and ratings of how much story context is needed. | Identifying event representations that support genuinely long-range comprehension rather than local lexical matching; the context-extent rating is a useful analogue for event span. | Public download. Source license is recorded per article. Keep all questions from one story in the same split to avoid stimulus leakage. |
| [ROCStories and Story Cloze](https://cs.rochester.edu/nlp/rocstories/) | 98,159 crowdsourced five-sentence everyday stories and 3,744 Story Cloze instances with human-authored correct and incorrect endings. | Training short event chains, likely transitions, causal/temporal coherence, and “what happens next” priors. | Free after a simple access form. Each sentence often approximates one event, but that is a dataset convention rather than a validated cognitive segmentation. CaTeRS adds explicit relations to a 320-story subset. |
| [TellMeWhy](https://stonybrooknlp.github.io/tellmewhy/) | 9,636 short narratives with 30,519 human-written why questions and free-form answers. It includes which story sentences helped answer each question, explicit/implicit-answer judgments, and a subset with three-rater validity and grammaticality scores. | Goal and causal inference tied to particular event sentences; the helpful-sentence labels can supervise links from described actions to supporting context. | Public JSON/CSV and Hugging Face releases. It is built on ROCStories, so deduplicate at the underlying story level when combining datasets. |
| [LitBank](https://github.com/dbamman/litbank) and [BookNLP](https://github.com/booknlp/booknlp) | Human annotations over approximately 2,000 words from each of 100 public-domain works: asserted events, entities, coreference, and quotation/speaker attribution, totaling 210,532 tokens. BookNLP supplies trained models and richer book-level outputs. | Building separate event-mention, character-continuity, speaker, and event-realis tracks before attempting boundary inference. | LitBank is CC BY 4.0 and includes original Gutenberg excerpts. Event mentions are not event instances or boundaries; several mentions may describe one event, and discourse can mention hypothetical or out-of-order events. |
| [FABLES](https://github.com/mungg/FABLES) | Five LLMs generated 130 summaries for 26 recent novels. Human readers labeled 3,158 atomic claims for faithfulness, supplied evidence/reasons, and commented on chronology, salience, omissions, and over-emphasis. | Directly evaluating AI-rater reliability and learning which proposed events or character states are faithful, omitted, misplaced, or over-weighted in long-story summaries. | Public MIT-licensed annotation repository, but the copyrighted source books are not included and must be obtained separately. This is an evaluation set, not a large training corpus. |
| [OpenAI summarize-from-feedback](https://github.com/openai/summarize-from-feedback) | 64,832 human pairwise summary preferences on Reddit TL;DR posts, plus Likert-style evaluation data on several quality axes. | Large-scale human preference pretraining for content selection and compression; useful as a weak salience prior before narrative-specific calibration. | Public data and code. Posts are mixed-domain and only some are personal narratives; preference for a summary is not a temporal event label. Do not use its learned reward as the final event-validity criterion. |

### 5.3 Large open books and audiobook reservoirs

The following resources contain little or no human event supervision. Their value is scale, clean local feature extraction, narrator/book diversity, editorial chapter structure, and the ability to generate explicitly provenance-tracked AI segmentations. They should be described as self-supervised or pseudo-labeled sources, not human validation data.

| Resource | Scale and structure | Best use | Access and caveats |
|---|---|---|---|
| [Project Gutenberg](https://www.gutenberg.org/) | More than 75,000 free ebooks, predominantly works unrestricted by U.S. copyright, often with human-authored chapter/section structure and multiple text formats. | Large `word2psy` reservoir for within-book novelty, chapter hierarchy, recurring entities, and public-domain stimulus selection. It also supplies source texts for several corpora below. | No registration or fee. Copyright status varies by jurisdiction and a small number of works are distributed by permission rather than public-domain status; consult each ebook's embedded license and remove Gutenberg trademark/license material only in accordance with its [terms](https://www.gutenberg.org/policy/license). |
| [LibriSpeech](https://www.openslr.org/12/) | About 1,000 hours of carefully segmented and aligned English audiobook speech, with transcripts, original LibriVox MP3s, original Gutenberg books, and metadata. | The most manageable large paired `aud2psy`/`word2psy` source for learning cross-modal alignment and narrator-invariant acoustic/semantic representations. | CC BY 4.0. ASR-style short segments interrupt long event continuity; reconstruct book/chapter order from metadata or original files before event modeling. |
| [Libri-Light](https://ai.meta.com/tools/libri-light/) | More than 60,000 hours of English LibriVox audiobook audio from over 7,000 speakers, with nested 600-hour, 6,000-hour, and 60,000-hour releases; most audio is unlabeled. | Self-supervised acoustic event representation, speaker/narrator robustness, and large-scale AI pseudo-segmentation after reassembling continuous source order. | Public research download. The main corpus lacks aligned transcripts and human event labels, so it cannot directly validate semantic boundaries. |
| [Multilingual LibriSpeech](https://www.openslr.org/94/) | Audiobooks and transcripts in eight languages; the published release includes about 44,500 hours of English and roughly 6,000 hours across German, Dutch, Spanish, French, Italian, Portuguese, and Polish. | Testing whether acoustic and semantic event mechanisms transfer across languages, narrators, and translations. | CC BY 4.0 and very large—the English archive is multi-terabyte. Download language- or book-level subsets and preserve language, speaker, book, and chapter identifiers. |
| [LibriTTS-R](https://www.openslr.org/141/) | A restored, higher-quality version of the 585-hour LibriTTS corpus, covering 2,456 speakers with corresponding text. | Prosody- and voice-sensitive `aud2psy` development where 24 kHz quality is preferable to LibriSpeech, plus narrator-robust alignment experiments. | Public download with the same non-restrictive basis as LibriTTS. It inherits short TTS/ASR-oriented segmentation and has no event judgments. |

### Recommended word/audio benchmark ladder

1. **Direct boundary benchmark:** Michelmann/Zenodo, ScriptPriming, and EventRecall, keeping individual human and AI raters separate.
2. **Paired processing benchmark:** Natural Stories for audio/text/reading-time alignment, then Le Petit Prince or Alice for neural convergence.
3. **Long written-narrative benchmark:** GECO for continuous human processing, SQuALITY/QuALITY for human content/comprehension, and FABLES for human evaluation of AI-derived narrative claims.
4. **Large weak supervision:** BookSum, NarrativeQA, NarraSum, ROCStories/Story Cloze, TellMeWhy, and LitBank.
5. **Scale and robustness:** LibriSpeech first; then targeted Libri-Light or Multilingual LibriSpeech subsets only after the event objective and pseudo-labeling protocol are stable.

The central transfer test should compare the same boundary model across text-only and audio-plus-text versions of a story. Improvements from `aud2psy` should survive control for punctuation, sentence endings, narrator pauses, and chapter breaks; otherwise the model may be detecting performance conventions rather than event-model updating.

## 6. Mapping data to modeling targets

| Modeling target | Most useful resources |
|---|---|
| Graded boundary probability and human disagreement | Kinetics-GEBD; Michelmann/Zenodo; Dryad; ScriptPriming |
| Boundary semantics: what changed | Kinetics-GEB+; ActivityNet Captions; Red Balloon situation-change codes; StudyForrest |
| Fine/coarse hierarchy | Dynamic Cognition Laboratory materials; Red Balloon; Breakfast Actions; TRIPOD |
| Reusable event types and duration priors | EPIC-KITCHENS; Breakfast Actions; YouCook2; Charades |
| Character, setting, interaction, and goal tracks | MovieGraphs; MovieNet; StudyForrest; STORIUM |
| Temporal, causal, and subevent relations | MAVEN-ERE; CaTeRS; GLUCOSE; ATOMIC 2020 |
| Event salience and memory consequences | Naturalistic Free Recall; EventRecall; Sherlock; TVSum; SumMe |
| Spoken/text boundary transfer | Michelmann GPT data; ScriptPriming; EventRecall; Natural Stories |
| Moment-by-moment reading/listening difficulty | Natural Stories; GECO; Le Petit Prince; Alice |
| Long-range narrative comprehension | NarrativeQA; QuALITY; SQuALITY; TellMeWhy |
| Multiscale written-event salience | BookSum; SQuALITY; NarraSum; SummScreen; MovieSum |
| Literary event/entity/speaker tracks | LitBank; BookNLP; NarrativeQA |
| AI-rater calibration | Michelmann GPT data; EventRecall human/GPT/LLaMA segmentations; FABLES |
| Neural convergence | Dryad infant/adult data; Narratives; Sherlock; StudyForrest; Le Petit Prince; Alice |
| Large audio/text self-supervision | LibriSpeech; Libri-Light; Multilingual LibriSpeech; Project Gutenberg |

## 7. Recommended evaluation split

A defensible first design would use different dataset classes for different purposes:

1. **Pretrain boundary mechanics** on Kinetics-GEBD/GEB+, ActivityNet Captions, and procedural datasets.
2. **Learn situation and relation heads** from MovieGraphs, MovieNet, MAVEN-ERE, CaTeRS, and GLUCOSE.
3. **Tune only a small fusion/calibration layer** on a subset of human narrative boundary data.
4. **Reserve entire stimuli and source series** from ScriptPriming, the Michelmann stories, Dryad, and selected Dynamic Cognition Laboratory materials for final testing.
5. **Evaluate consequences separately** with recall, importance, eye tracking, or neural state changes. These measures should not be merged into the boundary gold label.
6. **Add a modality-transfer test** in which the same story is evaluated as text and aligned speech, with text-only, audio-only, and late-fusion ablations.

Splits should be by whole movie, story, book, author, narrator, recipe, episode, and preferably source collection—not by clips, chapters, or passages randomly drawn from the same work. If the same public-domain work appears in Gutenberg, LibriSpeech, BookSum, LitBank, NarrativeQA, or an imaging corpus, all versions should share a split. This prevents memorized plot or author style from masquerading as event inference. Narrator-held-out and author-held-out results should be reported separately when estimating acoustic and semantic generalization.

For YouTube-derived data, record retrieval date, original identifier, checksum, and missing-video status so later failures can be distinguished from model failures. For written and spoken narratives, record edition/translation, narrator, chapter, tokenization, transcript version, forced-alignment method, and whether editorial chapter or sentence boundaries were exposed to the model.

## 8. A common annotation schema

To make these resources interoperable, preserve the original labels and map them into a long-form table with at least:

```text
dataset
stimulus_id
source_work_id
modality
language
edition_or_translation
narrator_or_reader
rater_type          # individual human, consensus, expert/editorial, AI
rater_id_or_model
label_family        # boundary, interval, importance, recall, relation, gaze, neural
granularity
onset
offset
time_unit
token_start
token_end
sentence_or_clause_id
audio_onset_seconds
audio_offset_seconds
presentation_order
narrated_time_order
label_or_description
strength_or_confidence
relation_source_event
relation_target_event
original_split
stimulus_available
license_or_terms
provenance_url
```

Do not collapse individual boundary times to a consensus too early. Per-rater data allow reliability-aware targets, temporal uncertainty kernels, rater-specific granularity estimates, and evaluation against the human ceiling. For reading time, gaze, and neural signals, retain participant-level measurements and nuisance variables rather than only grand averages.

AI annotations should retain model version, prompt, decoding settings, date, repeated samples, source-text window, and whether chapter titles or summaries were visible. They should never leak into a nominally human-held-out test set. Editorial paragraph, sentence, and chapter boundaries should also be marked explicitly so that their contribution can be measured rather than silently supplied as ground truth.
