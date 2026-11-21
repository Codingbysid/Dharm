# Decision Corpus v1.0

108 passage records in three equal collections, 216 modern applications, 108 misapplications and 42 English evaluation dilemmas. The 24-field exports are equivalent after decoding their three nested CSV columns. All records await specialist review. The v0.2 filenames are retained for existing loaders.

## Benchmark results

The v0.2.3 retrieval function was run afresh at every cumulative checkpoint. Across the final 38 answerable dilemmas, macro Recall@3 was **72.37%**. Both directions were correct on **4 of 14 pairs**. It abstained correctly on two empirical questions and attempted to answer the other two.

| Index | Records | Queries | Macro Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 36 | 18 | 96.875% | 6/6 | 2/2 | 2/2 | 3/4 |
| A | 54 | 24 | 86.364% | 5/8 | 2/2 | 2/2 | 3/6 |
| B | 72 | 30 | 83.929% | 5/10 | 2/2 | 2/2 | 4/8 |
| C | 90 | 36 | 73.529% | 4/12 | 2/2 | 2/2 | 5/10 |
| D | 108 | 42 | 72.368% | 4/14 | 2/2 | 2/4 | 5/10 |

### Original 18 questions against each index

| Index | Records | Queries | Macro Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 36 | 18 | 96.875% | 6/6 | 2/2 | 2/2 | 3/4 |
| A | 54 | 18 | 96.875% | 5/6 | 2/2 | 2/2 | 2/4 |
| B | 72 | 18 | 90.625% | 5/6 | 2/2 | 2/2 | 2/4 |
| C | 90 | 18 | 78.125% | 4/6 | 2/2 | 2/2 | 2/4 |
| D | 108 | 18 | 81.25% | 4/6 | 2/2 | 2/2 | 2/4 |

The expanded index fails the requested zero-regression criterion. On the final index Q01 and Q09 lost their correct arbitration primaries. Q14 retained both relevant passages in the top three but placed D038 first. Q16 lost the partial credit it earned on the original index. Adding documents changes inverse document frequencies, the leading candidates, graph expansion and the corpus-calibrated abstention floor.

For callers requiring exact v0.2.3 behavior, build the index over the first 36 records. The reproduction below supports this explicit version choice before reading any query. The compatibility number applies only to that index.

### New questions alone

| Queries | Macro Recall@3 | Correct flips | Abstention precision | Abstention recall | Ordered synthesis |
| --- | --- | --- | --- | --- | --- |
| Q19–Q42 on 108 records | 65.909% | 0/8 | undefined (0 predictions) | 0/2 | 3/6 |

None of the eight new contrast pairs had the correct primary on both variants. Q25/Q26 and Q31/Q32 illustrate the difference between finding the relevant pair and switching its primary. Q41 and Q42 require a physical assay and an unavailable historical observation respectively. Both elicited passage IDs.

### Evaluation protocol

Recall@3 averages the fraction of each answerable query’s distinct gold passages among the first three predictions before the gate. A paired query has one gold passage. A synthesis query has two. Flip accuracy counts a pair only when both top-one answers match their respective primary and both pass the gate. Abstention precision divides correct abstentions by predicted abstentions. Recall divides them by gold abstentions. Ordered synthesis requires the two gold passages at ranks one and two in the specified order after the gate. Each fraction reports successes over its own denominator.

Only dilemma_text enters prediction. Gold IDs, distractors, guidance and query group are evaluation metadata. Q01–Q18 were previously used to diagnose and revise this baseline. The new dilemmas are separately written from the passage examples, using the planned contrast and workflow labels. They form an authored diagnostic set with known target relationships. Independent, blinded test estimates would require a separately collected set. No scorer parameters were fitted to the expansion queries.

### Decisions by query

A, B, C and D show the first result after the gate at each checkpoint. The last two columns show the initial lexical seeds and final ranking on all 108 passages before the gate. An em dash means the query had not yet been added.

| Query | Group/type | Gold in order | A | B | C | D | Final seeds | Final top three |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Q01 | P1 | D001 | D001 | D001 | D042 | D042 | D042, D074, D073 | D042, D074, D073 |
| Q02 | P1 | D010 | D010 | D010 | D042 | D010 | D042, D073, D001 | D010, D042, D073 |
| Q03 | P2 | D007 | D007 | D007 | D007 | D007 | D007, D080, D010 | D007, D018, D080 |
| Q04 | P2 | D023 | D023 | D023 | D023 | D023 | D084, D007, D010 | D023, D084, D007 |
| Q05 | P3 | D015 | D015 | D015 | D015 | D015 | D017, D073, D049 | D015, D017, D050 |
| Q06 | P3 | D017 | D017 | D017 | D017 | D017 | D017, D049, D082 | D017, D082, D049 |
| Q07 | P4 | D025 | D025 | D025 | D025 | D025 | D025, D026, D035 | D025, D026, D035 |
| Q08 | P4 | D026 | D026 | D026 | D026 | D026 | D025, D026, D035 | D026, D025, D035 |
| Q09 | P5 | D035 | D037 | D057 | D057 | D057 | D057, D079, D038 | D057, D079, D037 |
| Q10 | P5 | D036 | D036 | D036 | D036 | D036 | D036, D079, D050 | D036, D079, D050 |
| Q11 | P6 | D003 | D003 | D003 | D003 | D003 | D003, D006, D033 | D003, D006, D033 |
| Q12 | P6 | D006 | D006 | D006 | D006 | D006 | D003, D006, D033 | D006, D003, D033 |
| Q13 | synthesis | D030 → D031 | D030 | D030 | D030 | D030 | D030, D023, D021 | D030, D031, D023 |
| Q14 | synthesis | D016 → D029 | D038 | D038 | D038 | D038 | D102, D029, D038 | D038, D016, D029 |
| Q15 | synthesis | D024 → D019 | D024 | D024 | D024 | D024 | D019, D024, D070 | D024, D019, D070 |
| Q16 | synthesis | D032 → D033 | D053 | D053 | D053 | D053 | D034, D054, D103 | D053, D034, D054 |
| Q17 | abstention | Abstain | Abstain | Abstain | Abstain | Abstain | D060, D040, D057 | D060, D040, D057 |
| Q18 | abstention | Abstain | Abstain | Abstain | Abstain | Abstain | D031, D075, D057 | D031, D055, D057 |
| Q19 | P7 | D039 | D039 | D039 | D039 | D039 | D039, D002, D063 | D039, D063, D002 |
| Q20 | P7 | D040 | D002 | D063 | D063 | D063 | D002, D063, D010 | D063, D002, D010 |
| Q21 | P8 | D049 | D049 | D049 | D049 | D049 | D049, D018, D026 | D049, D018, D026 |
| Q22 | P8 | D050 | D049 | D049 | D049 | D049 | D049, D010, D018 | D049, D010, D018 |
| Q23 | synthesis | D040 → D041 | D040 | D040 | D040 | D040 | D040, D004, D064 | D040, D041, D004 |
| Q24 | synthesis | D049 → D051 | D051 | D051 | D051 | D051 | D097, D092, D051 | D051, D098, D097 |
| Q25 | P9 | D057 | — | D057 | D057 | D057 | D085, D057, D108 | D057, D085, D108 |
| Q26 | P9 | D058 | — | D057 | D057 | D057 | D058, D057, D016 | D057, D058, D016 |
| Q27 | P10 | D067 | — | D072 | D072 | D072 | D068, D060, D093 | D072, D067, D068 |
| Q28 | P10 | D068 | — | D067 | D090 | D090 | D089, D067, D068 | D090, D089, D067 |
| Q29 | synthesis | D064 → D065 | — | D064 | D064 | D064 | D064, D065, D088 | D064, D065, D088 |
| Q30 | synthesis | D066 → D062 | — | D062 | D062 | D062 | D062, D043, D029 | D062, D043, D029 |
| Q31 | P11 | D085 | — | — | D085 | D085 | D019, D085, D051 | D085, D019, D051 |
| Q32 | P11 | D086 | — | — | D085 | D085 | D086, D047, D019 | D085, D086, D047 |
| Q33 | P12 | D089 | — | — | D036 | D036 | D089, D090, D035 | D036, D023, D035 |
| Q34 | P12 | D090 | — | — | D090 | D090 | D090, D035, D089 | D090, D036, D023 |
| Q35 | synthesis | D073 → D075 | — | — | D073 | D073 | D073, D076, D075 | D073, D075, D076 |
| Q36 | synthesis | D082 → D083 | — | — | D082 | D082 | D082, D076, D105 | D082, D076, D105 |
| Q37 | P13 | D093 | — | — | — | D094 | D094, D018, D093 | D094, D018, D093 |
| Q38 | P13 | D094 | — | — | — | D094 | D094, D093, D018 | D094, D093, D018 |
| Q39 | P14 | D105 | — | — | — | D002 | D002, D048, D033 | D002, D048, D033 |
| Q40 | P14 | D106 | — | — | — | D074 | D033, D074, D048 | D074, D033, D048 |
| Q41 | abstention | Abstain | — | — | — | D041 | D041, D017, D060 | D041, D017, D060 |
| Q42 | abstention | Abstain | — | — | — | D013 | D013, D015, D040 | D013, D015, D040 |

### Batch A checkpoint detail

| Query | Seeds | Expanded pool | Top three before gate | After gate |
| --- | --- | --- | --- | --- |
| Q01 | D042, D001, D021 | D001, D004, D010, D021, D027, D028, D038, D041, D042 | D001, D042, D021 | D001 |
| Q02 | D042, D001, D021 | D001, D004, D010, D021, D027, D028, D038, D041, D042 | D010, D042, D001 | D010 |
| Q03 | D007, D018, D010 | D001, D007, D010, D013, D018, D023, D029 | D007, D018, D010 | D007 |
| Q04 | D007, D010, D045 | D001, D007, D010, D013, D018, D023, D043, D045, D046 | D023, D007, D018 | D023 |
| Q05 | D017, D049, D051 | D015, D017, D049, D050, D051, D052 | D015, D017, D049 | D015 |
| Q06 | D017, D049, D040 | D015, D017, D039, D040, D041, D049, D050, D051 | D017, D049, D040 | D017 |
| Q07 | D025, D026, D035 | D012, D023, D025, D026, D028, D035, D036 | D025, D026, D035 | D025 |
| Q08 | D025, D026, D035 | D012, D023, D025, D026, D028, D035, D036 | D026, D025, D035 | D026 |
| Q09 | D038, D035, D009 | D009, D020, D022, D023, D035, D036, D037, D038, D042 | D037, D035, D038 | D037 |
| Q10 | D036, D050, D038 | D034, D035, D036, D037, D038, D042, D049, D050, D054 | D036, D050, D037 | D036 |
| Q11 | D003, D033, D006 | D003, D006, D014, D022, D032, D033 | D003, D033, D006 | D003 |
| Q12 | D003, D006, D033 | D003, D006, D014, D022, D032, D033 | D006, D003, D033 | D006 |
| Q13 | D030, D023, D021 | D007, D021, D023, D027, D028, D030, D031, D035 | D030, D031, D023 | D030 |
| Q14 | D029, D038, D031 | D016, D018, D029, D030, D031, D034, D037, D038, D042 | D038, D016, D029 | D038 |
| Q15 | D019, D024, D036 | D005, D019, D024, D030, D034, D035, D036 | D024, D019, D036 | D024 |
| Q16 | D034, D032, D054 | D004, D009, D032, D033, D034, D036, D050, D053, D054 | D053, D034, D032 | D053 |
| Q17 | D040, D017, D053 | D015, D017, D039, D040, D041, D052, D053, D054 | D040, D017, D053 | Abstain |
| Q18 | D031, D019, D032 | D005, D009, D018, D019, D024, D030, D031, D032, D033 | D031, D019, D032 | Abstain |
| Q19 | D002, D039, D010 | D001, D002, D010, D011, D013, D037, D039, D040 | D039, D002, D010 | D039 |
| Q20 | D002, D010, D021 | D001, D002, D010, D011, D013, D021, D027, D028 | D002, D010, D021 | D002 |
| Q21 | D049, D026, D018 | D007, D018, D025, D026, D028, D029, D049, D050, D051 | D049, D026, D018 | D049 |
| Q22 | D049, D026, D010 | D001, D010, D013, D025, D026, D028, D049, D050, D051 | D049, D010, D026 | D049 |
| Q23 | D040, D004, D041 | D001, D004, D016, D039, D040, D041, D042 | D040, D041, D004 | D040 |
| Q24 | D051, D040, D048 | D039, D040, D041, D044, D047, D048, D049, D051, D052 | D051, D040, D048 | D051 |

## Corpus structure and validation

D001–D036 and Q01–Q18 were preserved byte-for-byte as file prefixes, including the original CSV header and quoting. Every new passage has two applied scenarios, a negative example, an application limit and eight operational cues. The original status vocabulary is retained. Reversibility describes the focal commitment, which can be irreversible even when the recommended review that precedes it is reversible. Urgency and source-family counts describe sampling choices.

| Batch | Passages | Questions | Vedic | Gita | Chanakya | One-way | Two-way | High | Medium | Low |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original | D001–D036 | Q01–Q18 | 12 | 12 | 12 | 10 | 26 | 7 | 14 | 15 |
| A | D037–D054 | Q19–Q24 | 6 | 6 | 6 | 6 | 12 | 5 | 6 | 7 |
| B | D055–D072 | Q25–Q30 | 6 | 6 | 6 | 7 | 11 | 4 | 7 | 7 |
| C | D073–D090 | Q31–Q36 | 6 | 6 | 6 | 6 | 12 | 4 | 7 | 7 |
| D | D091–D108 | Q37–Q42 | 6 | 6 | 6 | 6 | 12 | 4 | 7 | 7 |
| All | D001–D108 | Q01–Q42 | 36 | 36 | 36 | 35 | 73 | 24 | 41 | 43 |

Read JSONL as UTF-8, with one object per line. In CSV use a quoted-field parser and decode scenarios_json, competing_passage_id and complementary_passage_id with json.loads. IDs are stable join keys. Keep gold labels and predictions outside the search documents.

## Schema (24 fields)

| Field | Type | Meaning |
| --- | --- | --- |
| id | string | Stable zero-padded ID, D001–D108. |
| collection | string | Source family, one of three collections. |
| work | string | Named text within the family. |
| locator | string | Edition-specific chapter and complete textual units. Disjoint selections are explicitly listed. |
| edition | string | Digital witness, translation provenance and variant notes. |
| source_url | string | Primary landing page for the cited witness. Ranges may require adjacent verse pages. |
| sanskrit_devanagari | string | Complete cited units in Devanagari. Verse numbers are carried in locator. |
| iast_transliteration | string | IAST aligned to the supplied Devanagari, with bars for daṇḍas. |
| english_translation | string | Full editorial English rendering of the selected units. |
| context_summary | string | Speaker, addressee and immediate argument. |
| derived_principle | string | Editorial decision heuristic used in retrieval. |
| decision_domain | string | Single primary category of the modern decision. |
| framework_type | string | Editorial framework for the application. |
| guna_or_posture | string | Source-specific or editorial classification. See interpretation notes below. |
| stakeholder_dynamic | string | Dominant relationship in the application. |
| urgency_level | string | Typical urgency of the modeled decision. |
| reversibility | string | Whether the focal application preserves the option to reverse the commitment. |
| scenarios_json | array | Array of exactly two objects, each with scenario and suggested_solution. |
| misapplication_scenario | string | One concrete harmful or invalid inference. Exclude from positive-only embedding text. |
| application_limit | string | Textual, ethical or epistemic boundary. |
| competing_passage_id | object | Object with id, condition, retain_self_when and flip_to_competing_when. Both directional fields describe observable evidence. |
| review_status | string | Legacy schema value draft_v0.2, pending_specialist_review. Applies across this release. |
| retrieval_cues | string | Eight on new rows. Frozen rows use eight to twelve. Comma-separated operational search terms drawn from the passage and its applications. |
| complementary_passage_id | object | id identifies a linked passage and sequence_role says whether it precedes or follows this stage, with a rationale. |

## Graph topology

At every checkpoint all targets resolve, there are no self-loops, and every passage has one incoming and one outgoing edge of each type. Each of the twelve new six-record blocks is closed. Number its six positions from 1 to 6. The reciprocal competing pairs are (1,2), (3,4), (5,6). The reciprocal workflow pairs are (1,3), (4,5), (6,2), with the first position in each pair supplying the first workflow stage. The two edge types have different targets on all 72 new records. The frozen corpus has seven existing exceptions to cross-type disjointness: D002, D005, D015, D017, D027, D030 and D033. The original competing graph also includes longer directed cycles.

## Editions and interpretation

Each record quotes a named unit or identifies its prose boundaries and gives a full editorial rendering. Vedic pitch and recitation features are generally omitted. IAST follows the displayed Devanagari. Digital witnesses sometimes join compounds or count verses differently. Chanakya Tradition is an editorial grouping of works of different genres and dates. Modern scenarios, directional edges and gold labels are editorial proposals. They do not establish historical intent, a policy’s effectiveness or a measured outcome.

Sattvic normally describes the applied target, except where a Gita passage explicitly discusses the qualities. D067, D075 and D077 contrast all three. D095 describes going beyond the qualities. Niti / Prudence covers non-diplomatic advice. Historical rules about social status, family descent, royal power and contracts require contextual reading. See each application_limit before using the corresponding scenario.

### Source choices

The proposed source list and the quota of six records from each collection per batch conflict in eight places. The decision contrasts were retained with the following substitutions. Some modern principles therefore rest on indirect analogies.

| ID | Selected unit | Scope of analogy |
| --- | --- | --- |
| D051 | Mundaka Upanishad 2.2.8 | The knot of bondage supplies an analogy for causal removal. Lifecycle cost is a modern check. |
| D070 | Rigveda 10.117.3 | Feeding a needy traveler supports prompt aid. Payment for completed work has an independent contractual basis. |
| D072 | Rigveda 1.164.11 | The recurring celestial wheel illustrates a reliable cycle. Operational deadlines need their own evidence. |
| D074 | Atharvaveda Śaunaka 3.24.2 | The grain-gathering hymn illustrates replenishing shared stores. It does not prescribe emergency taxation. |
| D084 | Bhagavad Gita 18.43 | Governance and firmness supply the analogy for acting within a delegated civilian role. |
| D088 | Rigveda 10.101.3 | Preparing a field illustrates creating productive capacity. Diversification is an extension. |
| D093 | Chandogya Upanishad 6.1.1 | A father sends a prospective heir of a learning tradition for instruction. The modern succession example is open to any qualified candidate. |
| D108 | Rigveda 10.191.3 | Collective counsel and offering provide a civilian analogy for agreeing a shared aim. |

Other locator choices follow the cited witness. D040 uses Arthaśāstra 1.12.15 for agreement among three reports. D048 uses Nīti 3.15 for the burning tree, D089 uses 5.2 for the gold tests, D098 uses 1.8 on residence, D103 uses 16.20 on book-bound learning and D106 uses 7.14 on circulating wealth. D083 uses Gita 13.8–12 in the cited numbering. D038 and D091 distinguish the two instructions by their exact prose boundaries in Taittirīya 1.11. D050 cites a complete stanza with refrain from a longer narrative. D037 addresses knowing the basis of sacred speech, D050 describes perseverance, and D053 concerns Krishna’s cosmic vision. Direct testing, a bounded sprint and early intervention are explicitly modern applications of these passages.

## Passage inventory

| Passage | Collection | Work and unit | Applied decision principle | Competing | Workflow |
| --- | --- | --- | --- | --- | --- |
| D001 | Vedas & Upanishads | Rigveda 10.191.2 | Record the constraints each party must meet. Test the options against the constraints they accept in common. | D010 | Precede with D004 |
| D002 | Vedas & Upanishads | Rigveda 10.117.5 | Determine how much is genuinely disposable and what documented need it could meet before assigning the surplus. | D011 | Follow with D011 |
| D003 | Vedas & Upanishads | Rigveda 10.34.13 | When repeated bets consume resources needed for an existing commitment, stop the bets and identify the work those resources must support. | D006 | Follow with D014 |
| D004 | Vedas & Upanishads | Atharvaveda 3.30.3 | Invite each willing party to describe the disputed behavior, its effect and one change they can both check. | D016 | Follow with D001 |
| D005 | Vedas & Upanishads | Shukla Yajurveda 40.6 | Before choosing for others, identify one assumption about their experience and test it with the people affected. | D008 | Follow with D008 |
| D006 | Bhagavad Gita | Bhagavad Gita 2.47 | Name the duty, the controllable next step and the evidence of careful execution before acting under uncertain reward. | D003 | Follow with D022 |
| D007 | Bhagavad Gita | Bhagavad Gita 3.21 | Before taking a visible shortcut, ask what would happen if everyone copied it. Demonstrate the practice you expect others to repeat. | D023 | Follow with D018 |
| D008 | Bhagavad Gita | Bhagavad Gita 18.63 | After giving advice, ask the decision owner to state the trade-off in their own words and choose freely. | D020 | Precede with D005 |
| D009 | Chanakya Tradition | Arthashastra 1.7.9 | Before committing others to a plan, obtain evidence from the people who know its dependencies and record what changed. | D022 | Precede with D020 |
| D010 | Chanakya Tradition | Arthashastra 1.19.30 | Check what will become impossible with delay. Give a credible expiring claim an immediate hearing and preserve a reversible remedy. | D001 | Follow with D013 |
| D011 | Chanakya Tradition | Arthashastra 1.19.34 | Identify who bears the cost of a service change and compare measured access before and after a reversible trial. | D019 | Precede with D002 |
| D012 | Chanakya Tradition | Chanakya Niti 12.22 | Choose a small repeatable contribution, record each addition and periodically test whether the cumulative result serves its purpose. | D014 | Precede with D025 |
| D013 | Vedas & Upanishads | Samaveda Uttararchika 1874 (Kauthuma 9.3.9.2) | When noise or alarm makes observation unreliable, take a brief, bounded pause and obtain one independently checkable observation. | D027 | Precede with D010 |
| D014 | Vedas & Upanishads | Katha Upanishad 1.2.1–1.2.2 | Before an exclusive commitment, state the end each option serves and which opportunity accepting it permanently forfeits. | D002 | Precede with D003 |
| D015 | Vedas & Upanishads | Brihadaranyaka Upanishad 5.2.1–5.2.3 | Identify the behavior producing the shortfall. Restrain overreach, release resources that are being withheld, or relieve a burden imposed on someone else. | D017 | Follow with D017 |
| D016 | Vedas & Upanishads | Mundaka Upanishad 3.1.6 | Before putting a consequential statement on the permanent record, distinguish what the evidence establishes, what remains uncertain and who needs to receive it. | D013 | Follow with D029 |
| D017 | Vedas & Upanishads | Isha Upanishad 1 | Before consuming a nonrenewable stock, establish who has a claim, what minimum use meets the purpose, and who can authorize that use. | D015 | Precede with D015 |
| D018 | Vedas & Upanishads | Rigveda 10.173.1–10.173.2 | Maintain a publicly stated service commitment through a change of leadership. Ask those who rely on it how they will detect a lapse. | D029 | Precede with D007 |
| D019 | Vedas & Upanishads | Atharvaveda 12.1.45 | When groups share a scarce service, compare actual access by language and custom, then test a common rule with accommodations. | D005 | Precede with D024 |
| D020 | Bhagavad Gita | Bhagavad Gita 2.7 | When your loyalties make it hard to weigh evidence, name the conflict and ask a qualified, disinterested person to examine the options. | D024 | Follow with D009 |
| D021 | Bhagavad Gita | Bhagavad Gita 2.14 | When a disturbance fits a known transient pattern, retain the plan until a stated review time. Define evidence that would trigger earlier intervention. | D028 | Precede with D027 |
| D022 | Bhagavad Gita | Bhagavad Gita 2.38 | Once authority, alternatives and consequences have been checked, separate fear of losing from any new evidence that would justify withholding a final commitment. | D033 | Precede with D006 |
| D023 | Bhagavad Gita | Bhagavad Gita 3.35 | For an exclusive commitment, compare the mandate, capabilities and obligations of the decision owner with those of the peer being imitated. | D007 | Follow with D035 |
| D024 | Bhagavad Gita | Bhagavad Gita 18.20–18.22 | Ask what the analysis treats as shared, what it separates, and whether it mistakes one observation for the whole. Test the model against a case that could disprove it. | D030 | Follow with D019 |
| D025 | Bhagavad Gita | Bhagavad Gita 6.5 | Choose one default under your control, change it for a fixed interval, and compare the resulting behavior with the prior pattern. | D026 | Follow with D012 |
| D026 | Bhagavad Gita | Bhagavad Gita 18.23–18.25 | Ask who authorized the work, what reward is driving its scope, who bears the cost and whether the people doing it have the capacity. Revise the proposal when these answers conflict. | D025 | Follow with D028 |
| D027 | Bhagavad Gita | Bhagavad Gita 18.30–18.32 | At an imminent decision, state the governing obligation and the evidence for acting or stopping. Check whether a local incentive has reversed the meaning of compliance. | D021 | Follow with D021 |
| D028 | Bhagavad Gita | Bhagavad Gita 18.37–18.39 | Compare how a choice feels now with what it enables after a review interval. Check whether the apparent relief merely obscures a continuing problem. | D012 | Precede with D026 |
| D029 | Chanakya Tradition | Arthashastra 2.10.6 and 2.10.57 | Before issuing a binding instruction, ask two recipients independently to identify the trigger, responsible party and required result. Resolve incompatible readings and duplicate obligations before signature. | D034 | Precede with D016 |
| D030 | Chanakya Tradition | Arthashastra 6.1.1 | Audit seven dependencies before expansion: decision authority, managers, people or market served, infrastructure, reserves, delivery capacity and partners. Require evidence and an owner for the weakest dependency. | D031 | Follow with D031 |
| D031 | Chanakya Tradition | Arthashastra 7.1.2 and 7.1.6–7.1.11 | Compare a negotiated pact, open competition, waiting, advancing, obtaining backing and different lawful positions with different partners. Document relative capability and which future options a commitment forecloses. | D018 | Precede with D030 |
| D032 | Chanakya Tradition | Arthashastra 9.1.13–14, 9.1.16 and 9.1.33 | Before adding staff or spending reserves, identify the information that could change the choice. Recheck implementation capacity, local conditions and the available window against that finding. | D009 | Follow with D033 |
| D033 | Chanakya Tradition | Arthashastra 9.4.1–9.4.3 | Separate lost operating capacity from cash and inventory consumed. Compare the remaining options using future costs, credible gains and the option to stop. | D032 | Precede with D032 |
| D034 | Chanakya Tradition | Arthashastra 2.10.47 | For a verified breach, document what dialogue and concessions achieved. Consider transparent separation of conflicting roles or authorized enforcement when a lesser remedy cannot protect the affected parties. | D004 | Follow with D036 |
| D035 | Chanakya Tradition | Chanakya Niti 1.11 | Use relevant, consensual evidence of follow-through to assign responsibility. Check what resources and authority the person actually had before inferring reliability. | D036 | Precede with D023 |
| D036 | Chanakya Tradition | Chanakya Niti 6.16–6.17 (variant witnesses) | Set observable readiness conditions and assign a decision owner. Once the conditions hold, concentrate the agreed effort on completing the chosen task. | D035 | Precede with D034 |
| D037 | Vedas & Upanishads | Rigveda 1.164.39 | Run a small independent test and record the result before accepting a portable claim. | D038 | Follow with D039 |
| D038 | Vedas & Upanishads | Taittiriya Upanishad 1.11, devapitṛkāryābhyām through end | Compare the reasoning of disinterested practitioners when a choice turns on a norm that cannot be tested in time. | D037 | Precede with D042 |
| D039 | Bhagavad Gita | Bhagavad Gita 3.40–41 | Identify the evaluator’s stake, record a prediction before seeing the result and commission an unconflicted review. | D040 | Precede with D037 |
| D040 | Chanakya Tradition | Arthashastra 1.12.15 | Trace an allegation to independent first-hand reports and record disagreements before taking action on it. | D039 | Follow with D041 |
| D041 | Vedas & Upanishads | Chandogya Upanishad 4.4.1–5 | Disclose the exact boundary between what is verified and what remains unknown when someone relies on your attestation. | D042 | Precede with D040 |
| D042 | Chanakya Tradition | Chanakya Niti 2.7 | Restrict a provisional plan to people who must implement it and define the release point in advance. | D041 | Follow with D038 |
| D043 | Bhagavad Gita | Bhagavad Gita 3.11 | Give both parties a measurable benefit and review whether each contribution still supports the other. | D044 | Follow with D045 |
| D044 | Chanakya Tradition | Arthashastra 2.8.1–4 | Reconcile entrusted receipts against independent records before continuing a disputed transfer. | D043 | Precede with D048 |
| D045 | Vedas & Upanishads | Atharvaveda Śaunaka 7.12.1–2 | Give the local group authority to try a bounded adaptation and record dissent before review. | D046 | Precede with D043 |
| D046 | Chanakya Tradition | Arthashastra 2.7.3 | Require a shared definition and reporting interval wherever units must reconcile the same figures. | D045 | Follow with D047 |
| D047 | Bhagavad Gita | Bhagavad Gita 18.48 | Continue a sound activity when measured imperfections stay within an agreed tolerance and correction is available. | D048 | Precede with D046 |
| D048 | Chanakya Tradition | Chanakya Niti 3.15 | Isolate the specific failed component immediately when a reproduced breach can spread across the system. | D047 | Follow with D044 |
| D049 | Bhagavad Gita | Bhagavad Gita 6.16–17 | Set a repeatable work-rest schedule when the same effort must be sustained across many cycles. | D050 | Follow with D051 |
| D050 | Vedas & Upanishads | Aitareya Brahmana 7.15, fifth stanza to Rohita and refrain | Concentrate available effort on a verified short opportunity with a defined end and recovery period. | D049 | Precede with D054 |
| D051 | Vedas & Upanishads | Mundaka Upanishad 2.2.8 | Replace the common cause after a controlled test shows it explains repeated failures and the lifetime cost favors removal. | D052 | Precede with D049 |
| D052 | Bhagavad Gita | Bhagavad Gita 5.20 | Use an existing buffer for an isolated disturbance and specify what recurrence would reopen the design. | D051 | Follow with D053 |
| D053 | Bhagavad Gita | Bhagavad Gita 11.33 | Act before a verified liability permanently destroys a remaining option and record the evidence for intervention. | D054 | Precede with D052 |
| D054 | Chanakya Tradition | Arthashastra 7.4.2–3 | Hold position while a named indicator improves the value of waiting and delay preserves the option. | D053 | Follow with D050 |
| D055 | Vedas & Upanishads | Kena Upanishad 2.1 | State what your present evidence cannot establish and suspend a confident claim until that gap is specified. | D056 | Follow with D057 |
| D056 | Bhagavad Gita | Bhagavad Gita 4.34 | Approach a qualified practitioner with one concrete question and check the explanation against a worked example. | D055 | Precede with D060 |
| D057 | Vedas & Upanishads | Taittiriya Upanishad 3.10, opening hospitality instruction through ya evaṃ veda | Offer baseline participation when access can be safely provided and no restricted resource is at issue. | D058 | Precede with D055 |
| D058 | Chanakya Tradition | Arthashastra 2.21.2 | Before granting restricted access, verify the minimum evidence of identity, scope and authorization. | D057 | Follow with D059 |
| D059 | Bhagavad Gita | Bhagavad Gita 12.15 | Reduce avoidable agitation when the communication has no corrective purpose or immediate consequence. | D060 | Precede with D058 |
| D060 | Bhagavad Gita | Bhagavad Gita 17.15 | Give a necessary correction promptly, privately when possible, with evidence and a practical remedy. | D059 | Follow with D056 |
| D061 | Chanakya Tradition | Arthashastra 7.2.21 | Seek bounded outside backing when neither of the available bilateral relationships can protect a vital service. | D062 | Follow with D063 |
| D062 | Chanakya Tradition | Arthashastra 7.3.2 | Negotiate a reciprocal settlement when both parties can deliver agreed terms without a protector. | D061 | Precede with D066 |
| D063 | Bhagavad Gita | Bhagavad Gita 16.1–3 | Raise a material concern with the affected decision-maker when silence would permit avoidable harm. | D064 | Precede with D061 |
| D064 | Chanakya Tradition | Chanakya Niti 7.1 | Keep personal details within a trusted reporting channel when public exposure adds no protective benefit. | D063 | Follow with D065 |
| D065 | Vedas & Upanishads | Rigveda 10.71.2 | Edit a crowded account down to the facts that actually distinguish the options, preserving dissent and provenance. | D066 | Precede with D064 |
| D066 | Chanakya Tradition | Arthashastra 1.15.34 | Convene a small council covering the missing expertise and give others a channel to submit evidence. | D065 | Follow with D062 |
| D067 | Bhagavad Gita | Bhagavad Gita 18.26–28 | Assess how the same person handles success, failure and responsibility before assigning greater discretion. | D068 | Follow with D069 |
| D068 | Chanakya Tradition | Arthashastra 1.8.28 | Test a claimed qualification using a representative, consented work sample with a published rubric. | D067 | Precede with D072 |
| D069 | Vedas & Upanishads | Prashna Upanishad 1.2–3 | Set a finite, supported preparation period before admitting a learner to work that requires prerequisite knowledge. | D070 | Precede with D067 |
| D070 | Vedas & Upanishads | Rigveda 10.117.3 | Release an available benefit promptly when the recipient’s immediate need is established and further testing serves no purpose. | D069 | Follow with D071 |
| D071 | Bhagavad Gita | Bhagavad Gita 4.18 | Check whether visible activity changes the relevant outcome, including the consequences of doing nothing. | D072 | Precede with D070 |
| D072 | Vedas & Upanishads | Rigveda 1.164.11 | Align work with a documented repeating cycle when moving the start changes the feasible result. | D071 | Follow with D068 |
| D073 | Chanakya Tradition | Arthashastra 4.3.17 | Deploy supplies to meet a verified immediate shortfall while protecting the means for the next recovery cycle. | D074 | Follow with D075 |
| D074 | Vedas & Upanishads | Atharvaveda Śaunaka 3.24.2 | Refill a common reserve when the current need is covered and the next replenishment is uncertain. | D073 | Precede with D078 |
| D075 | Bhagavad Gita | Bhagavad Gita 17.20–22 | Match a gift to a recipient’s actual need, with suitable timing and no repayment condition. | D076 | Precede with D073 |
| D076 | Vedas & Upanishads | Rigveda 1.125.5 | Offer a general benefit without asking recipients to prove a return when scarcity and misuse risks are low. | D075 | Follow with D077 |
| D077 | Bhagavad Gita | Bhagavad Gita 18.33–35 | Continue a feasible commitment while its purpose, evidence and effects remain sound; review what makes you persist. | D078 | Precede with D076 |
| D078 | Chanakya Tradition | Arthashastra 7.17.1–2 | Offer a bounded concession when it secures a credible settlement and preserves the parties’ essential interests. | D077 | Follow with D074 |
| D079 | Vedas & Upanishads | Maitri Upanishad 6.34, stanza beginning mana eva manuṣyāṇām | Observe what repeatedly captures attention and loosen that attachment before adding external controls. | D080 | Follow with D081 |
| D080 | Chanakya Tradition | Arthashastra 1.6.1 | Put a procedural check between a predictable personal impulse and a decision over other people’s interests. | D079 | Precede with D084 |
| D081 | Vedas & Upanishads | Atharvaveda Śaunaka 6.64.2 | Agree a shared objective and a verifiable common commitment before distributing interdependent work. | D082 | Precede with D079 |
| D082 | Chanakya Tradition | Arthashastra 9.7.5 | Subtract acquisition costs, reversals and third-party exposure before accepting a seemingly profitable transfer. | D081 | Follow with D083 |
| D083 | Bhagavad Gita | Bhagavad Gita 13.8–12 | Invite correction and identify whose knowledge is missing before closing a disputed diagnosis. | D084 | Precede with D082 |
| D084 | Bhagavad Gita | Bhagavad Gita 18.43 | Use delegated authority to assign a temporary safe course when a live disruption makes further consultation costly. | D083 | Follow with D080 |
| D085 | Vedas & Upanishads | Shvetashvatara Upanishad 2.10 | Remove a recurring environmental disturbance before asking people to compensate by effort. | D086 | Follow with D087 |
| D086 | Bhagavad Gita | Bhagavad Gita 6.26 | When the setting is adequate, mark the current step and deliberately return to it each time attention drifts. | D085 | Precede with D090 |
| D087 | Chanakya Tradition | Arthashastra 2.16.6 | Reject a lucrative term if its verified effects undermine essential access or impose unacceptable harm on users. | D088 | Precede with D085 |
| D088 | Vedas & Upanishads | Rigveda 10.101.3 | Prepare independent supply capacity when the current bottleneck is a missing or fragile source of production. | D087 | Follow with D089 |
| D089 | Chanakya Tradition | Chanakya Niti 5.2 | Seek evidence across conduct, competence, generosity and completed work before granting an irreversible trust. | D090 | Precede with D088 |
| D090 | Bhagavad Gita | Bhagavad Gita 9.30–31 | Allow a supervised route back when the person accepts responsibility and shows sustained change. | D089 | Follow with D086 |
| D091 | Vedas & Upanishads | Taittiriya Upanishad 1.11, opening through svādhyāyapravacanābhyām | Refuse to certify a material falsehood even when an easier method would meet the deadline. | D092 | Follow with D093 |
| D092 | Bhagavad Gita | Bhagavad Gita 2.50 | Improve the method while keeping the underlying obligation and truthful account intact. | D091 | Precede with D096 |
| D093 | Vedas & Upanishads | Chandogya Upanishad 6.1.1 | Give a prospective successor supervised responsibility and assess readiness before the transfer is due. | D094 | Precede with D091 |
| D094 | Chanakya Tradition | Arthashastra 5.6.33 | Activate a time-limited deputy under the existing succession mandate when the office suddenly becomes vacant. | D093 | Follow with D095 |
| D095 | Bhagavad Gita | Bhagavad Gita 14.22–25 | Observe fluctuating approval and blame without changing a sound course until an operational criterion moves. | D096 | Precede with D094 |
| D096 | Bhagavad Gita | Bhagavad Gita 3.25 | Take a visible coordinating action when others depend on your example to sustain a common service. | D095 | Follow with D092 |
| D097 | Vedas & Upanishads | Rigveda 10.85.42 | Honor a durable mutual commitment when consent, safety and the shared purpose still hold. | D098 | Follow with D099 |
| D098 | Chanakya Tradition | Chanakya Niti 1.8 | Leave an arrangement when essential supports have disappeared and feasible remedies cannot restore them. | D097 | Precede with D102 |
| D099 | Bhagavad Gita | Bhagavad Gita 17.16 | Pause a reactive response and check motive before communicating an accusation. | D100 | Precede with D097 |
| D100 | Chanakya Tradition | Arthashastra 1.21.3 | Assign an authorized custodian to contain a verified physical access failure and record the handover. | D099 | Follow with D101 |
| D101 | Vedas & Upanishads | Brihadaranyaka Upanishad 1.4.14 | Apply an independent rule and an accessible appeal when unequal power prevents a valid claim from being heard. | D102 | Precede with D100 |
| D102 | Chanakya Tradition | Arthashastra 3.1.2 | Check authority, consent and the transaction record before treating an apparent agreement as binding. | D101 | Follow with D098 |
| D103 | Chanakya Tradition | Chanakya Niti 16.20 | Request a practical demonstration when a decision depends on knowledge being available at the moment of use. | D104 | Follow with D105 |
| D104 | Bhagavad Gita | Bhagavad Gita 4.38–39 | Build understanding through repeated practice with feedback when the learner has time to mature. | D103 | Precede with D108 |
| D105 | Vedas & Upanishads | Atharvaveda Śaunaka 3.24.1 | Gather a reserve while supply is abundant when later essential demand is predictable. | D106 | Precede with D103 |
| D106 | Chanakya Tradition | Chanakya Niti 7.14 | Release excess resources into productive use once the reserve floor and obligations are covered. | D105 | Follow with D107 |
| D107 | Bhagavad Gita | Bhagavad Gita 18.45–46 | Deepen competence in the assigned role when the shared aim is clear and a specific craft limits quality. | D108 | Precede with D106 |
| D108 | Vedas & Upanishads | Rigveda 10.191.3 | Agree on a shared aim and each group’s contribution when skilled people are optimizing incompatible outcomes. | D107 | Follow with D104 |

## Integrity fingerprints

| File | Bytes | SHA-256 |
| --- | --- | --- |
| decisions_v0.2.csv | 305933 | 75884bbddd3ba40328795ac39e604396c11756413be8174bc52cbc9b29eb96bb |
| decisions_v0.2.jsonl | 349131 | 18d35df30acf268c76088f477e328688bd56220d18abb0cbf9b8de4ec260043f |
| eval_queries_v0.2.jsonl | 23801 | 6a284a3250bba9ad08c5aee9a0ce8ec0032290ab8982758726498a9ba6bd96e9 |

## Reproducing the checkpoints

Python 3.12.14 and scikit-learn 1.8.0 were used for this run. Save the following block as benchmark.py next to the two JSONL downloads and run python benchmark.py 108. Supply 36, 54, 72 or 90 for the earlier checkpoints. Selection of a checkpoint happens before query scoring. The immutable build23 implementation is included with the historical baseline functions it shares. The wrapper selects a release prefix and reports the legacy subset separately.

~~~python
import sys, json, re
size=int(sys.argv[1]) if len(sys.argv)>1 else 108
assert size in (36,54,72,90,108)
query_count=18+(size-36)//3
from pathlib import Path
from statistics import quantiles
from sklearn.feature_extraction.text import TfidfVectorizer

def load(name):
    return [json.loads(line) for line in Path(name).read_text(encoding='utf-8').splitlines()[:query_count if name.startswith('eval') else size]]
import copy
release = load('decisions_v0.2.jsonl')
release23 = copy.deepcopy(release)
prior_cues = {'D019': 'unequal access, languages, translation, shared resource, subgroup comparison, usable hours, accommodation, service outcomes', 'D024': 'linked outcomes, isolated metric, shared process, data silos, partial view, cohort, cross check, interactions'}
for r in release:
    if r['id'] in prior_cues: r['retrieval_cues'] = prior_cues[r['id']]
records = copy.deepcopy(release)
historical = {'D003': ['repeated losses, prize chasing, chance reward, diverted supplies, existing obligation, reserved materials, sunk stake, opportunity cost', 'Another attempt only offers a chance reward and consumes supplies reserved for an existing obligation.', 'The work independently satisfies an existing commitment regardless of the outcome.'], 'D006': ['existing obligation, authorized work, uncertain reward, controllable effort, submit, completion, follow up, check eligibility', 'The task fulfills an existing obligation or independently worthwhile goal despite an uncertain reward.', 'Repeated attempts consume committed resources solely for a chance at recovery.'], 'D025': ['routine, default, follow through, time available, habit, workflow friction, planned start, personal control', 'Measured effort fits the available hours and a controllable routine still stalls completion.', 'The assigned workload exceeds available hours or depends on an unsafe burden.'], 'D026': ['workload, capacity, shift hours, staffing, scope, collateral burden, excessive strain, motive', 'Work exceeds staffed capacity or its design transfers foreseeable costs to others.', 'A timed check shows the work fits available hours and a personal default delays completion.']}
for r in records:
    if r['id'] in historical:
        r['retrieval_cues'],r['competing_passage_id']['retain_self_when'],r['competing_passage_id']['flip_to_competing_when']=historical[r['id']]
ids = [r['id'] for r in records]
edges = [ids.index(r['competing_passage_id']['id']) for r in records]
def vectorizer():
    return TfidfVectorizer(ngram_range=(1,2), stop_words='english', sublinear_tf=True)
def doc(r, full=False, scenarios=True):
    fields = [r[k] for k in ('english_translation','context_summary','derived_principle')]
    if full:
        fields += [r[k] for k in ('derived_principle','decision_domain')]
        if scenarios:
            fields.extend(s[k] for s in r['scenarios_json'] for k in ('scenario','suggested_solution'))
        fields += [r['application_limit'],r['retrieval_cues']]
    return ' '.join(fields)
def index(docs):
    v = vectorizer()
    return v, v.fit_transform(docs)
old = index([doc(r) for r in records])
new = index([doc(r,True) for r in records])
def sims(ix,q):
    v,m = ix
    return (v.transform([q]) @ m.T).toarray()[0]
# Calibration uses corpus scenarios queried against documents with examples removed.
cal = index([doc(r,True,False) for r in records])
self_scores = [sims(cal,s['scenario'])[i] for i,r in enumerate(records) for s in r['scenarios_json']]
floor = max(0.01,quantiles(self_scores,n=10,method='inclusive')[0])
def normalize(t):
    t=t.lower()
    # Represent the direction of negation before English stop-word removal.
    tags=[]
    rules=[
        (r'no delegated authority|lacks? (?:the |a )?(?:mandate|authority)|without (?:the )?authority','mandate mismatch'),
        (r'same (?:written |delegated )?authority|both actors have authority','common mandate'),
        (r'replenished|replacement is secured|reserve already covers','renewable surplus'),
        (r'irreplaceable|replacement cannot','finite stock'),
        (r'minutes|imminent|closes before|expires today','expiring window'),
        (r'next month|stays open|remains open','open window'),
        (r'exceeds available hours|exceeds staffed capacity','over capacity'),
        (r'fits (?:the )?available hours','within capacity'),
    ]
    for pattern,term in rules:
        if re.search(pattern,t): tags.append(term)
    words='zero one two three four five six seven eight nine ten eleven twelve'.split()
    numeric=re.sub(r'\b('+'|'.join(words)+r')\b',lambda m:str(words.index(m[0])),t)
    h=re.search(r'(\d+(?:\.\d+)?)\s*[- ]?hours?\b.*?\b(?:of|within|against)\s+(?:an?\s+)?(\d+(?:\.\d+)?)\s*[- ]?hours?',numeric)
    if h: tags.append('over capacity' if float(h[1])>float(h[2]) else 'within capacity')
    return t+' '+' '.join(tags)
trigs=[normalize(r['competing_passage_id'][k]) for r in records for k in ('retain_self_when','flip_to_competing_when')]
tr=index(trigs)
def gate(q,score,threshold=floor):
    q=q.lower()
    action=bool(re.search(r'\b(should|choose|decid\w*|select|approve|commit|prioriti\w*|allocat\w*|recommend|authorize|reconsider|proceed|withdraw|ought|what comes first|what sequence|how (?:should|can|do))\b',q))
    empirical=bool(re.search(r'\b(provenance|which (?:source|site|origin)|which \w+ supplied|true (?:value|reading|temperature)|exact (?:value|reading|temperature)|to \w+ decimal places?|diagnos(?:e|is)|physical origin)\b',q))
    return (empirical and not action) or (score<threshold and not action)
def predict(q):
    a=sims(old,q); b=sims(new,q)
    def order(score):
        return sorted(range(len(ids)),key=lambda i:(-score[i],ids[i]))
    aorder=order(a); seeds=order(b)[:2]
    pool=set(seeds+[edges[i] for i in seeds])
    t=sims(tr,normalize(q))
    support=t[::2]-t[1::2]
    # Transfer evidence on outgoing edges. A neighbor's own margin votes against
    # its source only when its edge actually refers back to that source.
    votes={i:[] for i in pool}
    for i in sorted(pool):
        j=edges[i]
        votes[i].append(support[i])
        if j in pool: votes[j].append(-support[i])
    adj=b.copy()
    for i,vs in votes.items(): adj[i]+=0.20*max(-1,min(1,sum(vs)/len(vs)))
    # Prioritize expanded candidates, fill remaining positions by full-index rank.
    bord=sorted(pool,key=lambda i:(-adj[i],ids[i]))+[i for i in order(b) if i not in pool]
    return ([ids[i] for i in aorder[:3]],max(a)<.12),([ids[i] for i in bord[:3]],gate(q,max(b)))
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
protected = set("no not without never none neither only same different before after within exceeds under over".split())
stoplist = sorted(ENGLISH_STOP_WORDS - protected)
def index22(texts):
    v = TfidfVectorizer(ngram_range=(1,2), stop_words=stoplist, sublinear_tf=True)
    return v,v.fit_transform(texts)

def structural_text(text):
    s=normalize(text)
    # A negated predicate should not also emit its affirmative structural tag.
    negative=r"\b(?:not|never|cannot|doesn't|doesn’t)\s+(?:fully\s+)?(?:satisfy|fulfil[l]?|meet)\b"
    pos=r"\b(?:satisfies|fulfi[l]+s|meets)\b.{0,45}\b(?:duty|obligation|commitment|deliverable|requirement)\b"
    if re.search(negative,s):
        s+=" unmet obligation"
    elif re.search(pos,s):
        s+=" fulfilled obligation"
    return s

def build22(data):
    positive=index22([doc(r,True) for r in data])
    triggers=index22([structural_text(r['competing_passage_id'][k]) for r in data
                      for k in ('retain_self_when','flip_to_competing_when')])
    comp=[ids.index(r['complementary_passage_id']['id']) for r in data]
    calibration=index22([doc(r,True,False) for r in data])
    scores=[sims(calibration,s['scenario'])[i] for i,r in enumerate(data) for s in r['scenarios_json']]
    floor22=max(.01,quantiles(scores,n=10,method='inclusive')[0])
    def rank(q):
        raw=sims(positive,q)
        initial=sorted(range(len(data)),key=lambda i:(-raw[i],ids[i]))
        seed=initial[:3]
        pool=set(seed+[edges[i] for i in seed]+[comp[i] for i in seed])
        sim=sims(triggers,structural_text(q))
        final=raw.copy()
        for i in sorted(pool):
            j=edges[i]
            delta=.20*(sim[2*i+1]-sim[2*i])
            if j in pool and delta>0:
                # All proposals read the same snapshot to avoid cycle amplification.
                final[j]=max(final[j],max(raw[i],raw[j])+delta)
        ordered=sorted(pool,key=lambda i:(-final[i],ids[i]))
        return ([ids[i] for i in ordered[:3]],gate(q,max(raw),floor22)),([ids[i] for i in seed],[ids[i] for i in sorted(pool)])
    return rank

def normalize23(t):
    t=t.lower()
    # Represent the direction of negation before English stop-word removal.
    tags=[]
    rules=[
        (r'no delegated authority|lacks? (?:the |a )?(?:mandate|authority)|(?:authority|mandate)[^.]{0,35}\blacks?\b|without (?:the )?authority','mandate mismatch'),
        (r'same (?:written |delegated )?authority|both actors have authority','common mandate'),
        (r'replenished|replacement is secured|reserve already covers|can be replenished','renewable surplus'),
        (r'irreplaceable|replacement cannot','finite stock'),
        (r'minutes|imminent|closes before|expires today|cutoff arrives before','expiring window'),
        (r'next month|stays open|remains open','open window'),
        (r'exceeds available hours|exceeds staffed capacity','over capacity'),
        (r'fits (?:the )?available hours','within capacity'),
    ]
    for pattern,term in rules:
        if re.search(pattern,t): tags.append(term)
    words='zero one two three four five six seven eight nine ten eleven twelve'.split()
    numeric=re.sub(r'\b('+'|'.join(words)+r')\b',lambda m:str(words.index(m[0])),t)
    h=re.search(r'(\d+(?:\.\d+)?)\s*[- ]?hours?\b.*?\b(?:of|within|against)\s+(?:an?\s+)?(\d+(?:\.\d+)?)\s*[- ]?hours?',numeric)
    if h: tags.append('over capacity' if float(h[1])>float(h[2]) else 'within capacity')
    return t+' '+' '.join(tags)

def structural23(text):
    s=normalize23(text)
    # A negated predicate should not also emit its affirmative structural tag.
    negative=r"\b(?:not|never|cannot|doesn't|doesn’t)\s+(?:fully\s+)?(?:satisfy|fulfil[l]?|meet)\b"
    pos=r"\b(?:satisfies|fulfi[l]+s|meets)\b.{0,45}\b(?:duty|obligation|commitment|deliverable|requirement)\b"
    if re.search(negative,s):
        s+=" unmet obligation"
    elif re.search(pos,s):
        s+=" fulfilled obligation"
    return s


def build23(data):
    def document(r, scenarios=True):
        return ' '.join([doc(r,True,scenarios),r['competing_passage_id']['condition'],
                         r['complementary_passage_id']['sequence_role']])
    positive=index([document(r) for r in data])
    calibration=index([document(r,False) for r in data])
    self_scores=[sims(calibration,s['scenario'])[i] for i,r in enumerate(data) for s in r['scenarios_json']]
    floor23=max(.01,quantiles(self_scores,n=10,method='inclusive')[0])
    comp=[ids.index(r['complementary_passage_id']['id']) for r in data]
    triggers=index22([structural23(r['competing_passage_id'][k]) for r in data
                      for k in ('retain_self_when','flip_to_competing_when')])
    def rank(q):
        raw=sims(positive,q)
        seed=sorted(range(len(data)),key=lambda i:(-raw[i],ids[i]))[:3]
        pool=set(seed+[edges[i] for i in seed]+[comp[i] for i in seed])
        sim=sims(triggers,structural23(q))
        own=sim[1::2]-sim[::2]
        final=raw.copy()
        for i in sorted(pool):
            j=edges[i]
            if j not in pool: continue
            if edges[j]==i:
                net=.5*(own[i]-own[j])
            elif i in seed[:2] and own[i]>.05 and own[j]<0:
                net=own[i]
            else:
                continue
            if net>.005: final[j]=max(final[j],max(raw[i],raw[j])+.25*net)
        for i in sorted(pool):
            c=comp[i]
            if c not in pool or c<i or min(raw[i],raw[c])<.030:continue
            role=data[i]['complementary_passage_id']['sequence_role']
            first,second=(i,c) if role.startswith('Follow with') else (c,i)
            top=max(raw[i],raw[c])
            final[first]=max(final[first],top+.015)
            final[second]=max(final[second],top+.010)
        ordered=sorted(pool,key=lambda i:(-final[i],ids[i]))
        return ([ids[i] for i in ordered[:3]],gate(q,max(raw),floor23)),([ids[i] for i in seed],[ids[i] for i in sorted(pool)])
    return rank

predict22=build22(release)
predict23=build23(release23)
# Scoring is separate from prediction. Only raw question text enters predict().
queries=load('eval_queries_v0.2.jsonl')
runs22=[predict22(q['dilemma_text']) for q in queries]
predictions=[(*predict(q['dilemma_text']),result) for q,(result,trace) in zip(queries,runs22)]
runs23=[predict23(q['dilemma_text']) for q in queries]
predictions=[(*p,run[0]) for p,run in zip(predictions,runs23)]
def metrics(col):
    hits=[]; pairs={}; tp=ap=positive=0
    for q,p in zip(queries,predictions):
        rank,ab=p[col]; typ=q['query_type']; ap+=ab
        if typ=='abstention':
            positive+=1;tp+=ab;continue
        gold={q['gold_primary_id']}
        if typ=='synthesis':gold.add(q['gold_competing_id'])
        hits.append(len(gold.intersection(rank))/len(gold))
        if typ=='arbitration_pair':
            pairs.setdefault(q['pair_group'],[]).append(not ab and rank[0]==q['gold_primary_id'])
    return [sum(hits)/len(hits),sum(all(v) for v in pairs.values()),tp,ap,positive]
if __name__=='__main__':
    for q, ((top, abstain), (seeds, pool)) in zip(queries,runs23):
        print(q['query_id'], 'Abstain' if abstain else top[0], top, seeds, pool)
    print('all: macro recall, successful pairs, correct/predicted/gold abstentions', metrics(3))
    queries, predictions = queries[:18], predictions[:18]
    print('original questions on selected index', metrics(3))
~~~
