# Training clinical agents to ask, test, commit and defer

*Manuscript draft. Every number comes from the generated reports in `docs/` (RESULTS.md and the JSON files named in each
section); figures are in `docs/figures/` as vector PDFs. The typeset version is `docs/paper/DEFER-Dx.pdf`
(`bash scripts/build_paper.sh`). Methods follow the code; `docs/METHODS.md` lists every constant.*

## Abstract

**Background.** Interactive diagnostic agents for acute abdominal pain are evaluated in a closed world: they must name
one of four diagnoses and cannot hand a case to a clinician. Abstention learned by reinforcement learning is expected to
collapse into refusal, and theory recommends thresholding a calibrated confidence instead. Meanwhile, a randomised study
has shown that physicians follow misdirecting AI advice, so what an agent hands over matters as much as when it does.

**Methods.** DEFER-Dx is a Qwen3-8B agent trained with multi-turn reinforcement learning to examine, order any of 22
priced tests, commit to a diagnosis with a stated probability, or defer to a clinician with a probabilistic
differential. Each deferral is credited by a counterfactual escalation value: continuations branched from the state where
the agent deferred, with deferral disabled, estimate the clinical value of continuing; the handoff is scored with a
strictly proper rule. We evaluated on 480 held-out MIMIC-IV-Ext-CDM admissions, on an open-world extension with five
time-critical diagnoses held out of training, and on MIMIC-CDM-CR, 32 cases rebuilt from recent published case reports,
with two training seeds for DEFER-Dx and for an identically trained agent without deferral.

**Results.** DEFER-Dx answered 77.6% of in-set cases at 95.2% accuracy (95% CI 93.4–96.8) with 3.8 unflagged errors per
100 cases (zero-shot: 24.1). Against the agent without deferral thresholded at matched coverage, it was more accurate on
answered cases in all four seed pairings (+1.1 to +5.3 points; pooled +3.2, 1.6–5.1). On 187 never-seen time-critical
admissions both seeds escalated 87% and committed wrongly in 6.8% and 7.1%, against 11.4% for an identically trained
deferral reward from prior work. A threshold sent complicated disease to the clinician less often than uncomplicated
disease (16.7% vs 24.0%); DEFER-Dx did not. But what the handoff named depended on the training seed: when clinicians
were modelled as adopting the handoff's top suggestion as often as physicians followed adversarial AI, one seed's
handoffs lowered team accuracy (86.8% vs 91.5% with the threshold) and the other's steered towards wrong in-scope
diagnoses on unfamiliar emergencies. On 32 published case reports, the second seed escalated every out-of-scope case and
the first was misled by two reported diagnostic traps.

**Conclusions.** Crediting escalation against the counterfactual of continuing trains stable, well-targeted deferral that
outperforms post-hoc thresholding and a published deferral reward. Whether escalation is safe depends on the content of
the handoff, which should be trained and evaluated as carefully as the decision to escalate.

## Introduction

A patient arrives in the emergency department with abdominal pain. An interactive diagnostic agent asks for the
examination, orders a CT and names a diagnosis. Every such agent published on the MIMIC-IV clinical decision-making
benchmark (MIMIC-CDM)[1] must name one of four conditions (appendicitis, cholecystitis, diverticulitis or
pancreatitis), even when the patient has a perforated ulcer or mesenteric ischaemia, and even when its own evidence is
split. None can hand the case back.

Three developments make this gap pressing. Regulators expect clinical decision support to leave the decision with a
clinician who can independently review the basis of its recommendations rather than rely on them[2], and a randomised
study of 225 physicians has just shown how fragile that safeguard is: conversational AI with a concealed objective raised
harmful decisions from 2% to 49%, and clinical expertise did not protect[3]. The field's leaderboard shows the cost of
optimising accuracy alone: the strongest reported MIMIC-CDM agent raises case-weighted accuracy to 93.4% while its
accuracy on the minority condition, diverticulitis, falls below that of a random test-ordering policy[4]. And recent
theory argues that the obvious fix, an "abstain" action trained by reinforcement learning, collapses into refusing
everything, and that one should instead threshold a calibrated confidence after training[5].

We ask whether an agent can learn when to defer without collapsing, whether learned deferral beats thresholding the
confidence of an identically trained agent, and what happens to the patient once the case reaches the clinician. Our
contributions:

1. **DEFER-Dx**, an agent trained with multi-turn group-relative policy optimisation over four actions: ask, order one
   of 22 individual tests, commit with a stated probability, or defer with a ranked probabilistic differential.
2. **The counterfactual escalation value (CEV).** At each state where the agent escalates, forced continuations branched
   from that exact state estimate the clinical value of not escalating. The escalation is credited with the difference,
   and the handoff is scored with a strictly proper rule. Prior deferral rewards price escalation by case-level
   difficulty[6–8] or by designated cases[9].
3. **Two open-world benchmarks.** MIMIC-CDM-OW, 713 admissions whose principal diagnosis is none of the four, built with
   MIMIC-CDM's own text pipeline, with five time-critical diagnosis groups held out of training; and MIMIC-CDM-CR, 32
   cases rebuilt from open-access case reports published 2020–2025, which can be shared without a data use agreement.
4. **A head-to-head test of learned versus post-hoc deferral** at exactly matched coverage, with two training seeds per
   arm, and a model of the clinician who receives the handoff, calibrated to the randomised evidence on AI advice[3].

## Results

### An agent that asks, tests, commits or defers

Each episode is one emergency admission (Fig. 1). The agent sees the history at presentation and, turn by turn, asks for
the examination, orders individual tests (each charged at its 2025 price), commits to one of the four conditions or
OTHER with a stated probability, or defers to a clinician with a probabilistic differential. Observations are looked up
from the record, never generated. DEFER-Dx was trained for 150 steps on 2,404 patient-disjoint training cases (CDM,
OTHER and same-pipeline in-set controls) with the counterfactual escalation value (Fig. 2; Methods) and a PI-controlled
cap of 30% on in-set deferral. A second training seed, an identically trained agent without DEFER (the GRPO control; two
seeds), a case-level group-consensus deferral reward (the mechanism of prior work[6–8]) and zero-shot Qwen3-8B were
evaluated on the same cases (Extended Data Fig. 1).

![Fig. 1 | DEFER-Dx. One admission per episode; the agent asks, orders individual priced tests, commits with a stated probability or defers with a probabilistic differential. The label space is open: the four MIMIC-CDM conditions or OTHER.](docs/figures/fig_overview.png)

![Fig. 2 | The counterfactual escalation value. When an episode ends in DEFER at state s, the environment is replayed to s and K = 3 forced continuations are sampled with escalation disabled (a DEFER executes as COMMIT of the differential's top choice; further tests remain allowed). Their mean clinical return V̂(s) values not escalating; the DEFER turn's advantage is E − V̂(s), where E values the escalation itself, including a strictly proper score of the handed-over differential.](docs/figures/fig_cev.png)

### Learned escalation without collapse

On the 480 held-out CDM admissions (val + test, three evaluation seeds), DEFER-Dx answered 77.6% (74.3–80.6) of cases at
95.2% (93.4–96.8) accuracy, with 3.8 (2.4–5.1) unflagged errors (wrong commitments without escalation) per 100 cases,
against 24.1 for zero-shot Qwen3-8B with forced choice and 9.1 with deferral offered in the prompt (paired −5.3, −7.5 to
−3.3). Its stated probabilities were calibrated (ECE 0.042; Brier 0.047, the lowest of all systems), and at 70% coverage
it was 96.5% accurate (Extended Data Fig. 3). Deferral did not collapse: the PI controller held in-set deferral near its
cap without the limit cycle that plain dual ascent produced (Extended Data Fig. 9); without the cap, it did (below). The second training seed answered
73.2% (69.7–76.5) at 95.3% (93.2–97.1) with 3.5 unflagged errors per 100. Against a confidence threshold on zero-shot
Qwen3-8B at matched coverage, DEFER-Dx was 12.4 points (9.4–15.7) more accurate on answered cases, and 11.1 (7.0–15.3)
with three-sample voting for both.

### Learned versus post-hoc deferral across training seeds

The fair comparison for learned deferral is a threshold on an identically trained agent[5]. The GRPO control was
trained exactly as DEFER-Dx but without DEFER, and its stated probability was thresholded at DEFER-Dx's coverage,
cross-fitted between val and test. Stated probabilities are coarse (a single value covers 14–17% of the control's
cases), so commitments tied at the cut were kept with a probability fitted to reach the target coverage (Methods);
without this, a plain cut overshot the target and overstated the advantage (+3.1 instead of +2.3 for the first seed).

DEFER-Dx was more accurate on the cases it answered in all four pairings of two DEFER-Dx seeds and two control seeds
(Fig. 3): +2.3 (0.1–4.6), +5.3 (3.0–7.7), +1.1 (−0.8 to 3.3) and +4.1 (1.9–6.5) points, three of four intervals excluding
zero. Pooled over the pairings, the difference was +3.2 points (1.6–5.1), with 2.4 fewer unflagged errors (1.1–3.9) and
2.1 fewer confident errors (1.0–3.3) per 100 cases and a lower area under the risk–coverage curve (−0.014, −0.031 to
−0.005); DEFER-Dx's calibration error was higher (ECE +0.036, 0.014–0.050). The margin depended mainly on the control's
seed: identically trained controls differed in how well their stated probabilities ranked their own errors (AURC 0.044
and 0.074), whereas DEFER-Dx's accuracy on answered cases was the same in both seeds (95.2% and 95.3%).

![Fig. 3 | Learned versus post-hoc deferral across training seeds. Each row compares one DEFER-Dx training seed with one seed of the identically trained GRPO control, thresholded at that DEFER-Dx seed's coverage (paired, same 480 cases, matched coverage). The last row pools the four pairs: cases are resampled with the training runs held fixed, so its interval reflects case sampling; the rows show the variation between runs. Positive values favour DEFER-Dx; filled markers have a 95% interval that excludes 0.](docs/figures/fig_seed_pairings.png)

### Unfamiliar, time-critical presentations

On 187 admissions from five time-critical diagnosis groups never seen in training (mesenteric ischaemia, perforated or
bleeding ulcer, ruptured aortic aneurysm, diabetic ketoacidosis, ectopic pregnancy), both DEFER-Dx seeds escalated 87.0%
of cases and committed to one of the four conditions in 6.8% (4.3–10.0) and 7.1% (4.3–10.3) (Fig. 4). Against the
identically trained group-consensus reward of prior work (11.4%), the differences were −4.6 (−8.0 to −1.4) and −4.3
(−7.7 to −1.1), and on 206 OTHER admissions from groups seen in training −7.0 (−9.9 to −4.4) and −6.6 (−9.9 to −3.6),
at the same accuracy on answered in-set cases (+1.1 and +1.2, intervals including zero). Without a threshold the
control committed wrongly in 15.0% (paired −8.2, −12.1 to −4.6). The thresholded control committed wrongly about as
rarely as DEFER-Dx (7.5%), but in a different way: it escalated 21.0% of these cases and committed to "none of the
four" in 64.3%. On 258 same-pipeline in-set controls DEFER-Dx named OTHER for 0.3%, so its open-world behaviour is not
a source shortcut.

![Fig. 4 | Open world. Outcomes on OTHER admissions from (a) diagnosis groups seen in training and (b) five time-critical groups never seen in training: wrong commitment to one of the four conditions, deferral, and naming OTHER. 95% case-level bootstrap intervals.](docs/figures/fig_open_world.png)

### Published case reports

On the 32 published cases (three evaluation seeds each), the trained agents recognised most out-of-scope presentations,
and the differences sat in the cases that had misled clinicians in the reports (Fig. 5). The second DEFER-Dx seed
escalated all 19 out-of-scope cases. The first escalated 17 and committed to an in-scope diagnosis on two reported
traps: a ruptured aortic aneurysm with migratory right iliac fossa pain and an Alvarado score of 9 (CR02), and a
heterotopic pregnancy with a viable intrauterine pregnancy on ultrasound (CR09); its wrong in-scope commitments were
8.8% of out-of-scope episodes (0–22.8). The thresholded control committed wrongly on CR09 (7.0%) and gave no answer in
21.1% of out-of-scope episodes: on the five cases whose deciding test lies outside the catalogue (acute porphyria,
Amanita and thallium poisoning, adrenal crisis, myocardial infarction without chest pain) it ordered tests until its
budget ran out instead of escalating. The consensus reward diagnosed cholecystitis in the gallstone ileus, whose CT
showed gas in the gallbladder (CR18), and zero-shot forced choice committed wrongly on CR09 and on the myocardial
infarction. On in-scope cases DEFER-Dx escalated complicated disease and never uncomplicated disease: 44.4% and 18.5%
of episodes on the nine complicated cases (perforated appendicitis with necrotising fasciitis, emphysematous and
gangrenous cholecystitis, diverticulitis with pylephlebitis), against none on the four uncomplicated cases. It was
correct on 69.2% and 87.2% of in-scope episodes, against 100% for the control. With 32 cases these results describe
behaviour; they are not precise rates.

![Fig. 5 | MIMIC-CDM-CR: behaviour on 32 published case reports. Majority outcome over three evaluation seeds for each case (rows) and system (columns): C, correct in-scope diagnosis; E, escalated (DEFER, or COMMIT of OTHER, which names no diagnosis to act on); W, wrong in-scope diagnosis; M, no majority. Rows are grouped by the five held-out time-critical groups, groups in no MIMIC cohort, a group seen in training, and in-scope cases with and without complications by the guideline grade the report supports. Cases and citations: benchmarks/casereports/.](docs/figures/fig_casereports.png)

### Complicated disease

DEFER-Dx names a diagnosis; it does not stage it, and a surgeon's management depends on the stage. We labelled each
in-set evaluation case as complicated or not from structured data only (admission ICD codes and admission laboratory
values), following the guideline used for each condition: WSES and AAST grading for appendicitis[10], WSES staging for
diverticulitis[11], the revised Atlanta classification for pancreatitis[12] and the Tokyo 2018 grades for
cholecystitis[13]. 114 of 480 cases were complicated (appendicitis 48 of 191, cholecystitis 21 of 130, diverticulitis 21
of 51, pancreatitis 24 of 108).

A confidence threshold escalated complicated disease less often than uncomplicated disease (16.7% vs 24.0%; −7.4,
−14.1 to −0.5): complicated presentations produce confident, often correct commitments. DEFER-Dx did not (23.7% vs
22.0%; +1.6, −6.1 to 9.8; second seed 28.1% vs 26.4%) (Extended Data Fig. 6). On the same complicated cases, at matched
overall coverage, DEFER-Dx escalated more than the thresholded control in all four seed pairings (+4.4 to +7.0 points;
+7.0, 0.6 to 13.7 for the first seeds) and more than the consensus reward (+7.6, 2.9 to 12.9; +12.0, 6.4 to 18.1), with
fewer unflagged errors on complicated cases in all four control pairings (−1.8 to −5.0 per 100). No system, however,
escalated complicated disease preferentially: none is severity-aware.

### What the handoff says, and the clinician who reads it

A handoff is AI advice. Barrit and colleagues randomised 225 physicians to AI that advocated harmful options or correct
ones and found 46.9 more harmful decisions per 100 targeted decision points with the misdirecting AI, with almost no
detection[3]; incorrect AI suggestions have similarly lowered radiologists' accuracy from 79.7% to 19.8%[14]. We therefore
examined what DEFER-Dx hands over and modelled the clinician who receives it (Methods): with probability f the clinician
adopts the handoff's top suggestion, otherwise decides independently with the accuracy of the MIMIC-CDM physicians
(0.89)[1]. A commitment to OTHER names no diagnosis to act on and was treated as a handoff suggesting OTHER.

The handoff's content depended on the training seed. 68.4% of the first seed's in-set handoffs ranked OTHER first (81% of
its in-set handoffs had a wrong top suggestion), against 4.2% for the second seed (32%) and 8.6% for the consensus reward.
On never-seen time-critical cases the pattern reversed: the first seed's handoffs had an in-scope top suggestion in 21%,
the second seed's in 84%. With an independent clinician (f = 0) every system's team accuracy on in-set cases was about
94% (Fig. 6). At the susceptibility observed under adversarial advocacy (f = 0.47), the first seed's team accuracy fell
to 86.8%, against 91.5% for the thresholded control (−4.7, −6.6 to −2.8), whereas the second seed's held (91.4%); on
never-seen time-critical cases the second seed's handoffs led to wrong in-scope diagnoses in 37.0%, against 11.0% for
the thresholded control and 25.9% for the consensus reward. A learning-to-defer rule on the control's confidence (defer
whenever the stated probability is below the clinician's accuracy[15,16]) was the safest on these terms (8.4%), at almost
twice the clinician workload (43% of in-set cases, against 23%).

![Fig. 6 | The clinician who receives the handoff. Team outcomes as a function of the probability f that the clinician adopts the handoff's top suggestion; otherwise the clinician decides independently with accuracy 0.89 (MIMIC-CDM physicians[1]). a, Team accuracy on in-set cases. b, Wrong in-scope diagnoses caused by the AI on never-seen time-critical cases (its own wrong commitments plus clinicians who adopt an in-scope suggestion). The vertical line marks the adversarial-advocacy effect measured by Barrit et al.[3]. A COMMIT of OTHER counts as a handoff suggesting OTHER.](docs/figures/fig_team.png)

### What drives the escalation credit

**The estimator.** We re-sampled K = 16 forced continuations from 100 in-set and 60 OTHER deferral states of each final
policy (Extended Data Fig. 7). The K = 3 estimate used in training gave the credit the same sign as K = 16 in 88% and 92%
of in-set states (K = 1: 83% and 88%; K = 8: 95% and 96%). Had it not been able to defer, the first seed's policy would
have been right in only 32% of its in-set deferral states, the second seed's in 67%: the first seed's deferrals were
better targeted, and 60% of them were credited against 37% of the second seed's.

**Test cost.** Removing the cost of tests ordered after the deferral changed the sign of the credit in 0% and 1% of
states, and forced continuations ordered a test in only 23% and 18% of cases. The early deferral (0.84 tests per case)
reflects the agent's inability to resolve these cases, not the price of investigating. [[No-cost ablation: DEFER-Dx
trained without the test-cost term, against DEFER-Dx at step 100.]]

**The clinician the credit assumes.** The escalation value credits a deferral with the value of a clinician who decides
correctly with probability τ = 0.85. If that clinician were right in 70% instead, 46% rather than 60% of the first seed's
in-set deferrals would be credited (30% rather than 37% for the second seed).

**Ablations at 100 training steps** (Extended Data Fig. 8). A constant deferral reward with the same OTHER value and
controller matched DEFER-Dx on accuracy on answered cases (−0.4, −2.3 to 1.5), unflagged errors and wrong commitments on
never-seen time-critical cases (10.2% vs 10.7%), but deferred less and rarely named OTHER in its handoffs (24.7% vs
89.0% of never-seen time-critical cases). Removing the handoff score (η = 0) left accuracy on answered cases unchanged
(−1.1, −2.8 to 0.6) but made handoffs less informative (−6.5 points, 3.5 to 9.3, in favour of the score), reduced
deferral (coverage 75.9% vs 68.1%) and increased false commitments on OTHER from groups seen in training (9.5% vs 6.3%);
OTHER-first handoffs were more common without the score (55.1% vs 35.1%), so they come with the counterfactual credit
itself.

**Without the cap** (ρ_max = 1; Extended Data Fig. 9), in-set deferral in training batches matched DEFER-Dx's for 40
steps (25% in both) and then rose to 69% (steps 41–50) and 95% (steps 51–100), while tests per episode fell from 1.8 to
0.1. With the coverage multiplier at zero, 96–99% of branched deferrals were credited (E ≈ 0.9–1.0 against V̂ ≈ 0.1),
against 55% under the controller in steps 51–100. The final policy answered 3.0% (1.9–4.4) of held-out in-set cases,
against 68.1% (64.4–71.6) for DEFER-Dx at step 100, ordered 0.12 tests per case (1.57) and almost never committed wrongly
on OTHER (0.0% and 0.5% for seen and never-seen groups, against 6.3% and 10.2%); its withheld guesses were right in 35% of
in-set cases (78%). The escalation value alone therefore does not bound how much is handed over: an assumed clinician is
worth more than the agent's forced continuation in almost every state. What it does is rank states by the value of not
escalating; the multiplier ν, held by the controller, is the price of a deferral that turns the ranking into a budget.
The credit decides which cases are escalated, the cap how many. [[DEFER-Dx with the handoff score weighted 1.0.]]

### Robustness and equity

With every sentence containing CDM's diagnosis mask removed, DEFER-Dx's accuracy on answered cases changed by −1.0
points (−2.0 to 0.0) and its coverage by +1.6 (−0.2 to 3.5). The pattern of recorded tests predicts the diagnosis in
MIMIC-CDM (macro AUROC 0.879 from recorded/not-recorded flags alone), but DEFER-Dx received a "not performed" reply in
only 12–17% of episodes, and its accuracy on answered cases was lower after one (89.7% vs 95.7%; second seed 88.5% vs
96.2%), as for every trained model: there is no sign that it profits from test availability. Escalation and unflagged
errors did not differ systematically by sex, race, insurance or language; patients aged 80 or older were escalated more
often (crude 28.8% and 36.9% for the two seeds), consistent with atypical presentations in older patients (Extended
Data Table 2). On LA-CDM's test split, counting handoffs through their top choice, DEFER-Dx reached 76.4% (71.1–81.0)
mean-class accuracy and its second seed 85.2% (80.8–89.3), against 82.7% for the control, 86.3% for the consensus reward
and 86.9% for zero-shot Qwen3-8B restricted to the four labels (LA-CDM reports 81.3% in its own closed-world
environment[17]; these are context, not head-to-head comparisons).

## Discussion

Learned deferral did not collapse. Crediting each escalation against what continuing from the same state would have
achieved produced an agent that answered three-quarters of in-set cases at 95% accuracy with fewer than four unflagged
errors per 100, escalated 87% of dangerous presentations it had never seen, and did both in two independent training
runs. It outperformed the deferral reward of prior work on unfamiliar emergencies in both runs, and outperformed a
threshold on an identically trained agent in all four seed pairings; how much it outperformed the threshold depended
mainly on the threshold, because identically trained agents differed in how well their confidence ranked their own
errors. Theory is right that a proper-scored confidence makes thresholding a strong baseline[5]; in our runs a learned
operating point was the more reliable of the two.

The more important finding concerns the handoff. A deferral agent is evaluated as if the clinician who receives the case
were independent of it. Under that assumption every system here reached the same team accuracy of about 94%. Physicians
are not independent of AI advice[3,14,18], and once the clinician can be steered, the content of the handoff dominates:
one DEFER-Dx run labelled uncertain in-set cases "none of the four", the other escalated unfamiliar emergencies with an
in-scope suggestion, and either pattern would have cost patients under the susceptibility measured in physicians. Both
runs were trained with the same strictly proper handoff score; at η = 0.3 it improved handoff content (the no-handoff
ablation) but did not pin down which kind of differential the agent learned. A threshold, by contrast, hands over its
own withheld guess, and a learning-to-defer rule hands over fewer guesses but doubles the clinician's workload.
Regulatory guidance asks that clinicians be able to review the basis of a recommendation independently[2]; our results
suggest that agents should be trained and evaluated against a clinician who might not.

Complicated disease is where a surgeon's decision matters most. A threshold sent it to the clinician less often than
uncomplicated disease, because complicated presentations look certain; DEFER-Dx did not, but neither did it seek
complicated disease out. Staging is a capability none of the agents has[19]: a clinically useful agent would name the
condition, grade it by the relevant guideline[10–13] and escalate by grade.

**Implications.** Learned deferral should be compared with a threshold on an identically trained model, at exactly
matched coverage, with more than one training run of each, since both the coarse confidences of language models and
seed-to-seed variation make unmatched or single-run comparisons easy to misread. The volume of escalation should be
set as a budget by the clinical service rather than left to the reward: a credit that assumes a competent clinician,
trained without a budget, handed over almost every case. Agents that defer should be evaluated in
the open world, on external cases, and with a model of the clinician who receives the case; the handoff should be
trained with stronger pressure towards the right content and assessed by clinicians.

**Limitations.** The data come from one centre and one organ system (abdominal pain at BIDMC). Labels for OTHER cases are
principal ICD codes, severity labels are proxies from codes and admission laboratory values, and the severity matrix and
some test prices are proxies (Methods). The environment reveals only what was recorded, so test availability is
informative, and it is static: a patient's condition does not evolve while tests are pending, and the agent cannot treat
and reassess, which real emergency care does. The clinician model is a sensitivity analysis calibrated to an
adversarial study, not a measurement with these handoffs; clinician ratings of the handoffs (a blinded rating sheet is in
the repository) and a user study are needed. Two training seeds per arm show seed-to-seed variation but cannot estimate
it, and the consensus reward has one seed. MIMIC-CDM-CR has 32 cases, too few for precise rates; it is an external check
of behaviour, not a validation study. No model here should be used for patient care.

## Methods

### Data

**MIMIC-IV-Ext-CDM 1.1**[1] has 2,400 emergency admissions with acute abdominal pain (957 appendicitis, 648 cholecystitis,
538 pancreatitis, 257 diverticulitis). We kept LA-CDM's 80/10/10 split[17], so its numbers are comparable on its test
set. When a test repeats within an admission, the earliest result is used. **MIMIC-CDM-OW** (this work; Extended Data
Fig. 1) selects MIMIC-IV 2.2 admissions with an abdominal chief complaint whose principal diagnosis falls in one of nine
non-CDM groups and that carry no CDM-condition code at any position. Each case is built with MIMIC-CDM's own text
pipeline (history and examination windows, radiology section filter, masking of the case's own diagnosis, inclusion rule,
earliest-value deduplication under a total row order); rebuilds are byte-identical. It yields 713 OTHER cases and 648
same-pipeline in-set controls, which remove the shortcut "built by the open-world pipeline, so OTHER". Cohorts are
patient-disjoint: 2,404 training cases (1,760 CDM, 279 OTHER, 365 controls), 212 development cases, 480 CDM evaluation
cases (val + test), 206 OTHER cases from four groups seen in training, 187 from five time-critical groups held out of
training entirely, and 258 in-set controls (Extended Data Table 1). Groups with fewer than 10 cases are reported only
pooled.

### Environment

The hidden state is the diagnosis y ∈ {appendicitis, cholecystitis, diverticulitis, pancreatitis, OTHER}. At reset the
agent sees the history at presentation. Actions: ASK(physical examination), free; TEST(x) for 22 individual tests (blood
count, metabolic panels, liver tests, lipase, amylase, CRP, lactate, coagulation, triglycerides, pregnancy test,
urinalysis, blood and urine cultures, CT, ultrasound, MRI, abdominal and chest radiographs, HIDA), each charged at its
2025 BIDMC standard charge where published (placeholders otherwise); COMMIT(d, p); and DEFER(q, reason) with a
probabilistic differential q. At most eight investigations and two malformed turns are allowed.

### Rewards and the counterfactual escalation value

A commitment earns R = α·1[d = y] − λ(p − 1[d = y])² − κ·C[y, d] − c·Σcost, with α = λ = 1 (the second term is the Brier
score), κ = 0.5 and C a 5 × 5 asymmetric time-to-harm severity matrix; c makes ordering every test cost one correct
diagnosis. For an episode that ends with DEFER at state s, the environment is rebuilt by deterministic replay and K = 3
continuations are sampled from s with escalation disabled (a DEFER is executed as COMMIT of the differential's top choice
at its stated probability; further tests remain allowed). Their mean return V̂(s), charging only tests ordered after s,
values not escalating. The escalation is valued E = V_τ + η·S(q, y) − μ − ν, where V_τ is the expected commitment reward of
a calibrated decision that succeeds with probability τ on in-set cases (1.2 on OTHER cases), S is the normalised
multi-class Brier score of the handed-over differential (strictly proper; η = 0.3), μ = 0.1 the handoff cost and ν the
coverage multiplier, the price of a deferral (without it, nearly every deferral is credited; Results). The DEFER turn's advantage is E − V̂(s); earlier turns keep the group-relative advantage. At most 32
deferrals per step are branched; others get no update on their DEFER turn. τ is annealed from 0.95 to 0.85 over the first
100 steps.

### Training

Qwen3-8B in thinking mode with LoRA (rank 16, α = 32) on every linear layer, trained with multi-turn GRPO on batches of
12 cases × 8 episodes at temperature 1.0 for 150 steps: advantages centred without standard-deviation normalisation,
token-level clipped surrogate (ε 0.2/0.28) with truncated importance weights against the sampler's log-probabilities,
AdamW, no KL term. Generation (vLLM) and training (PEFT) share one 48 GB GPU. In-set deferral is held at or below 30%
by a PI controller on ν (K_p = 1.5, K_i = 0.2, ν_max = 2); plain dual ascent wound up and limit-cycled (Extended Data
Fig. 9). The control arm is identical without DEFER; the comparator arm replaces the escalation value with the
case-level consensus reward γ(τ − p̂) − μ (γ = 2), where p̂ is the share of the group's correct commitments (identical
to TIAR at τ = 0.5[6]). Second seeds change only the seed (case order, sampling, LoRA initialisation).

### Comparators and post-hoc thresholds

Each forced-choice system was thresholded on its stated probability at DEFER-Dx's coverage, fitted on val and applied to
test and vice versa. Because stated probabilities are coarse, commitments tied at the cut are kept with probability
k = (target × N − n_above)/n_tied, decided by a fixed hash of the case and evaluation seed; we report one pre-specified
draw and the range of the point difference over 20 draws (RESULTS.md §5b). Also evaluated: SGR (5% selective risk,
δ = 0.05)[22], self-consistency over three samples, and a learning-to-defer rule that defers whenever the stated
probability is below the clinician's accuracy, the Bayes-optimal rejector for a calibrated model and an expert of known
accuracy[15,16,21].

### Statistics

Every number is given with a 95% case-level bootstrap interval (2,000 resamples, all evaluation seeds of a case resampled
together); comparisons are paired bootstraps on the same cases with two-sided p-values. Pooled seed comparisons pool the
four pairings with cases resampled and training runs held fixed. The paper's two questions, fixed before the second
training seeds were run, give four primary tests: pooled accuracy on answered cases and pooled unflagged errors against
the thresholded control, and wrong commitments on never-seen time-critical cases against the consensus reward for each
DEFER-Dx seed. Holm correction within these four gives p ≤ 0.02 for each. Across all 218 paired tests in the report (one
family; two-sided Wald p from the bootstrap standard error; RESULTS.md §5), the pooled accuracy difference has Holm
p = 0.052, the unflagged-error difference 0.078 and the time-critical comparisons ≥ 0.80, which therefore rest on their
replication across two seeds; all other comparisons are exploratory.

### Severity labels

Complicated disease was defined from the admission's ICD-9/10 codes and the first admission laboratory values in the case
record (no note text): appendicitis with gangrene, perforation, abscess or generalised peritonitis; diverticulitis with
perforation, abscess, peritonitis, fistula or obstruction; pancreatitis with necrosis or pseudocyst, organ-failure codes
or creatinine > 1.9 mg/dL; cholecystitis with Tokyo 2018 grade II (WBC > 18 × 10⁹/L; perforation, fistula,
pericholecystic or liver abscess, peritonitis) or grade III (organ-failure codes; creatinine > 2.0 mg/dL, INR > 1.5,
platelets < 100 × 10⁹/L) (`scripts/severity_analysis.py`).

### The clinician model

For an escalated case (DEFER, or COMMIT of OTHER) the clinician adopts the handoff's top suggestion with probability f
and otherwise decides correctly with probability a_c = 0.89, the mean accuracy of the four physicians in the MIMIC-CDM
reader study[1]. Episodes without a usable answer carry no suggestion. On OTHER cases we count only AI-caused wrong
in-scope diagnoses (the AI's own wrong commitments, and clinicians adopting an in-scope suggestion), because the only
measured physician accuracy is for the four-way task. f ranged from 0 to 0.5; 0.47 is the adversarial-minus-aligned
difference in harmful decisions per targeted decision point in Barrit et al.[3] (`scripts/team_analysis.py`).

### Estimator audit

From the deferral states recorded in each final policy's held-out evaluation (100 in-set, 60 OTHER, sampled at
random), K = 16 forced continuations were sampled with the trainer's own branching function and training-time sampling,
and each continuation's return was stored by component; ν is the run's final value (`scripts/cev_audit.py`,
`scripts/cev_audit_report.py`).

### MIMIC-CDM-CR

Europe PMC was searched for open-access case reports with full text published 2020 or later, by target diagnosis;
reports with adult patients presenting with abdominal pain were kept, and reports with an ambiguous principal diagnosis,
in-hospital post-operative presentations or children were excluded. Each case section was paraphrased into the case
format (history with past history, medications and social history; examination; laboratory values at presentation in
MIMIC-IV units; imaging findings and impression); tests outside the catalogue were left out and noted, and nothing was
added. 19 cases are out of scope (7 from the held-out time-critical groups, 11 from groups in no MIMIC cohort, 1 bowel
obstruction) and 13 in scope with their guideline grade (9 complicated). The cases, citations and licences are in
`benchmarks/casereports/`; the texts are released under CC BY 4.0 with attribution to the source reports.

### Ethics

MIMIC-IV is de-identified and was used under the PhysioNet credentialed data use agreement. No MIMIC text was sent to
any third-party service: all models ran on local weights. Published case reports were used under their open-access
licences; MIMIC-CDM-CR contains no patient data beyond what the reports published.

## Data and code availability

Code, configurations, aggregate results, figures and MIMIC-CDM-CR: github.com/GIND123/Clinical-Agent-Stanford. MIMIC-IV,
MIMIC-IV-Note and MIMIC-IV-Ext-CDM are available to credentialed users from PhysioNet; MIMIC-CDM-OW is rebuilt
deterministically from them by the released code. Model weights trained on MIMIC will be released only through
PhysioNet's credentialed channel.

## References

1. Hager P, Jungmann F, Holland R, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. *Nat Med* 2024;30:2613–2622. doi:10.1038/s41591-024-03097-1
2. US Food and Drug Administration. Clinical Decision Support Software: guidance for industry and FDA staff. Revised January 2026.
3. Barrit S, Salvagno M, Corriero A, et al. Subversion of clinical judgment by conversational artificial intelligence. *medRxiv* 2026. doi:10.64898/2026.10.05.26364541
4. Shen et al. LDTL. arXiv:2604.05116, 2026.
5. Che et al. Abstention as an action can kill both the reward gradient and the KL anchor. arXiv:2608.00301, 2026.
6. Pan et al. TIAR. arXiv:2605.25850, 2026.
7. Gao et al. KARL. arXiv:2604.22779, 2026.
8. Zhang et al. AWA-RL. arXiv:2607.10738, 2026.
9. Zhan et al. TrustMed-RL. arXiv:2610.04387, 2026.
10. Di Saverio S, Podda M, De Simone B, et al. Diagnosis and treatment of acute appendicitis: 2020 update of the WSES Jerusalem guidelines. *World J Emerg Surg* 2020;15:27. doi:10.1186/s13017-020-00306-3
11. Sartelli M, Weber DG, Kluger Y, et al. 2020 update of the WSES guidelines for the management of acute colonic diverticulitis in the emergency setting. *World J Emerg Surg* 2020;15:32. doi:10.1186/s13017-020-00313-4
12. Banks PA, Bollen TL, Dervenis C, et al. Classification of acute pancreatitis—2012: revision of the Atlanta classification and definitions by international consensus. *Gut* 2013;62:102–111. doi:10.1136/gutjnl-2012-302779
13. Yokoe M, Hata J, Takada T, et al. Tokyo Guidelines 2018: diagnostic criteria and severity grading of acute cholecystitis. *J Hepatobiliary Pancreat Sci* 2018;25:41–54. doi:10.1002/jhbp.515
14. Dratsch T, Chen X, Rezazade Mehrizi M, et al. Automation bias in mammography: the impact of artificial intelligence BI-RADS suggestions on reader performance. *Radiology* 2023;307:e222176. doi:10.1148/radiol.222176
15. Mozannar H, Sontag D. Consistent estimators for learning to defer to an expert. *ICML* 2020.
16. Dvijotham KD, Winkens J, Barsbey M, et al. Enhancing the reliability and accuracy of AI-enabled diagnosis via complementarity-driven deferral to clinicians. *Nat Med* 2023;29:1814–1820. doi:10.1038/s41591-023-02437-x
17. Bani-Harouni et al. LA-CDM. *ICLR* 2026. arXiv:2506.13474.
18. Goddard K, Roudsari A, Wyatt JC. Automation bias: a systematic review of frequency, effect mediators, and mitigators. *J Am Med Inform Assoc* 2012;19:121–127. doi:10.1136/amiajnl-2011-000089
19. Nori H, Daswani M, Kelly C, et al. Sequential diagnosis with language models. arXiv:2506.22405, 2025.
20. Johri S, Jeong J, Tran BA, et al. An evaluation framework for clinical use of large language models in patient interaction tasks. *Nat Med* 2025. doi:10.1038/s41591-024-03328-5
21. Madras D, Pitassi T, Zemel R. Predict responsibly: improving fairness and accuracy by learning to defer. *NeurIPS* 2018.
22. Geifman Y, El-Yaniv R. Selective classification for deep neural networks. *NeurIPS* 2017.

## Extended Data

![Extended Data Fig. 1 | Data and evaluation design. Two sources and patient-disjoint cohorts. MIMIC-CDM-OW, built with MIMIC-CDM's own text pipeline, supplies OTHER cases and same-pipeline in-set controls; the five time-critical OTHER groups are held out of training entirely.](docs/figures/fig_data.png)

![Extended Data Fig. 2 | One training step. Multi-turn GRPO with the counterfactual escalation value: rollouts, forced continuations from deferral states, group-relative advantages for every turn and the escalation credit for DEFER turns, and the PI controller on the coverage multiplier.](docs/figures/fig_training.png)

![Extended Data Fig. 3 | Risk–coverage on CDM val + test (480 cases, 3 seeds). a, Accuracy on answered in-set cases as the stated-probability threshold is lowered; each curve ends (filled marker) at the system's own coverage. Hollow markers: forced-choice systems with a post-hoc threshold cross-fitted at the reference coverage (vertical line), ties at the cut broken to reach it exactly. b, Area under the risk–coverage curve.](docs/figures/fig_risk_coverage.png)

![Extended Data Fig. 4 | Paired differences against each comparator on the same cases. Circles: the first DEFER-Dx seed; diamonds (first two panels): the second seed, against the control thresholded at that seed's own coverage. Positive values favour DEFER-Dx (sign flipped for lower-is-better metrics, ↓); filled points have an interval that excludes 0.](docs/figures/fig_paired_differences.png)

![Extended Data Fig. 5 | Clinical safety on CDM val + test: unflagged errors, confident errors (wrong commitments stated with p > 0.8), severity-weighted error from the time-to-harm matrix, and tests ordered per case.](docs/figures/fig_clinical_safety.png)

![Extended Data Fig. 6 | Escalation by severity. a, Share of complicated (filled) and uncomplicated (hollow) in-set cases escalated; b, unflagged errors per 100 cases. Complicated disease defined from admission ICD codes and laboratory values by the guideline for each condition (Methods). 95% case-level bootstrap intervals.](docs/figures/fig_severity.png)

![Extended Data Fig. 7 | The escalation-value estimator. a, Share of in-set deferral states whose credit has the same sign with K forced continuations as with K = 16 (2,000 random subsets per state). b, Share of in-set deferrals credited when the escalation value assumes a clinician of the given accuracy; vertical lines: τ = 0.85 used in training, and 0.89, the MIMIC-CDM physicians.](docs/figures/fig_cev_audit.png)

![Extended Data Fig. 8 | Ablations at 100 training steps: DEFER-Dx's step-100 checkpoint against the same training with a constant deferral reward, without the scored handoff (η = 0), without the test-cost term, without the coverage cap, and with the handoff score weighted 1.0. Two evaluation seeds; 95% intervals.](docs/figures/fig_ablations.png)

![Extended Data Fig. 9 | Training dynamics and the coverage controller: in-set deferral against the 30% cap, the coverage multiplier ν, and development-set accuracy and coverage, for plain dual ascent (first run; ν wound up and limit-cycled), the PI controller (DEFER-Dx), the same training without the cap (ρ_max = 1; ν stays at zero and in-set deferral rises to about 95% after step 50) and the case-level consensus reward.](docs/figures/fig_training_dynamics.png)

![Extended Data Fig. 10 | Reliability of stated probabilities on committed CDM val + test cases (bins with at least 10 commitments).](docs/figures/fig_reliability.png)

![Extended Data Fig. 11 | Accuracy at full coverage by condition, CDM val + test; a deferral counts through the top of its handed-over differential.](docs/figures/fig_per_class.png)

![Extended Data Fig. 12 | Never-seen time-critical OTHER groups: wrong commitments and deferral by diagnosis group (groups with at least 10 cases; ruptured aneurysm and ectopic pregnancy pooled).](docs/figures/fig_unseen_groups.png)

**Extended Data Table 1 | Cohorts.** RL training 2,404 (1,760 CDM, 279 OTHER, 365 controls); development 212; CDM evaluation 480 (val 240, test 240; 51 diverticulitis); OTHER from groups seen in training 206; OTHER from five time-critical groups held out of training 187; same-pipeline in-set controls 258; MIMIC-CDM-CR 32. All cohorts are patient-disjoint.

**Extended Data Table 2 | Escalation equity by subgroup.** Escalation, accuracy on answered cases and unflagged errors per 100 cases by sex, age group, race, insurance and language, crude and label-adjusted, with 95% intervals, for both DEFER-Dx seeds, the consensus reward, the control and prompted zero-shot deferral (docs/fairness/; cells with fewer than 10 cases suppressed).

**Supplementary Data | MIMIC-CDM-CR.** The 32 cases, their source reports (authors, journal, DOI, PMID, PMCID, licence) and the construction protocol (benchmarks/casereports/).
