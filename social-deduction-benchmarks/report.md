# Research Report

> Benchmarking multi-agent LLM systems playing social deduction games (Mafia, Secret Hitler, Werewolf) using DSPy + ReAct, with memory, planning, observations, and game actions as tools

**Items:** 28  |  **Field categories:** 9

---

## Table of Contents

1. [Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf (Xu et al., 2023)](#1-exploring-large-language-models-for-communication-games-an-empirical-study-on-werewolf-xu-et-al-2023) — Year: 2023
2. [Hoodwinked: Deception and Cooperation in a Text-Based Game for Language Models](#2-hoodwinked-deception-and-cooperation-in-a-text-based-game-for-language-models) — Year: 2023
3. [Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game (Microscopic Analysis on LLM Players via Social Deduction Game)](#3-fine-grained-and-thematic-evaluation-of-llms-in-social-deduction-game-microscopic-analysis-on-llm-players-via-social-deduction-game) — Year: 2024
4. [Learning to Discuss Strategically: A Case Study on One Night Ultimate Werewolf](#4-learning-to-discuss-strategically-a-case-study-on-one-night-ultimate-werewolf) — Year: 2024
5. [Murder-Mystery Benchmarks: WhodunitBench (NeurIPS 2024) and ThinkThrice / 'Deciphering Digital Detectives' Jubensha (ACL 2024 Findings)](#5-murder-mystery-benchmarks-whodunitbench-neurips-2024-and-thinkthrice--deciphering-digital-detectives-jubensha-acl-2024-findings) — Year: 2024
6. [Strategic-Game Meta-Benchmarks: DSGBench, GTBench, GAMA-Bench, AgentBench](#6-strategic-game-meta-benchmarks-dsgbench-gtbench-gama-bench-agentbench) — Year: 2024
7. [Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction (Google, 2024)](#7-werewolf-arena-a-case-study-in-llm-evaluation-via-social-deduction-google-2024) — Year: 2024
8. [Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies (WereBench / WereAlign)](#8-beyond-survival-evaluating-llms-in-social-deduction-games-with-human-aligned-strategies-werebench--werealign) — Year: 2025
9. [Community LLM Werewolf/Mafia Arenas — MafiaBench and Foaster.ai Werewolf Arena](#9-community-llm-werewolfmafia-arenas--mafiabench-and-foasterai-werewolf-arena) — Year: 2025
10. [DVM: Towards Controllable LLM Agents in Social Deduction Games](#10-dvm-towards-controllable-llm-agents-in-social-deduction-games) — Year: 2025
11. [LSPO: Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization](#11-lspo-learning-strategic-language-agents-in-the-werewolf-game-with-iterative-latent-space-policy-optimization) — Year: 2025
12. [MaKTO — Multi-agent Kahneman-Tversky Optimization (Werewolf/Mafia)](#12-makto--multi-agent-kahneman-tversky-optimization-werewolfmafia) — Year: 2025
13. [MultiMind: Enhancing Werewolf Agents with Multimodal Reasoning and Theory of Mind](#13-multimind-enhancing-werewolf-agents-with-multimodal-reasoning-and-theory-of-mind) — Year: 2025
14. [The MindGames Challenge: Theory-of-Mind and Game Intelligence in LLM Agents (NeurIPS 2025 Competition)](#14-the-mindgames-challenge-theory-of-mind-and-game-intelligence-in-llm-agents-neurips-2025-competition) — Year: 2025
15. [The Traitors: Deception and Trust in Multi-Agent Language Model Simulations](#15-the-traitors-deception-and-trust-in-multi-agent-language-model-simulations) — Year: 2025
16. [Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games (Eckhaus et al., 2025)](#16-time-to-talk-llm-agents-for-asynchronous-group-communication-in-mafia-games-eckhaus-et-al-2025) — Year: 2025
17. [Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning](#17-training-language-models-for-social-deduction-with-multi-agent-reinforcement-learning) — Year: 2025
18. [WOLF: Werewolf-based Observations for LLM Deception and Falsehoods](#18-wolf-werewolf-based-observations-for-llm-deception-and-falsehoods) — Year: 2025
19. [Who's the Impostor? Multi-Agent Social Deduction for Evaluating LLM Social Reasoning (The Impostor Game)](#19-whos-the-impostor-multi-agent-social-deduction-for-evaluating-llm-social-reasoning-the-impostor-game) — Year: 2025
20. [Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games (Kao et al., 2026)](#20-hidden-in-plain-text-measuring-llm-deception-quality-against-human-baselines-using-social-deduction-games-kao-et-al-2026) — Year: 2026
21. [AmongAgents — Evaluating Large Language Models in the Interactive Text-Based Social Deduction Game (Among Us)](#21-amongagents--evaluating-large-language-models-in-the-interactive-text-based-social-deduction-game-among-us) — Year: 2024 (arXiv July 2024; presented at the Wordplay workshop, ACL 2024)
22. [AvalonBench / Avalon's Game of Thoughts (AGoT) — Recursive Contemplation (ReCon) for The Resistance: Avalon](#22-avalonbench--avalons-game-of-thoughts-agot--recursive-contemplation-recon-for-the-resistance-avalon) — Year: 2023 (both AvalonBench and Avalon's Game of Thoughts/ReCon; ReCon published at ACL 2024 Findings)
23. [DSPy agent stack — ReAct & memory (mem0) tutorials, optimizers (MIPROv2 / GEPA / BootstrapFewShot)](#23-dspy-agent-stack--react--memory-mem0-tutorials-optimizers-miprov2--gepa--bootstrapfewshot) — Year: DSPy v2 2023; DSPy 2.x with ReAct, MIPROv2, GEPA 2024-2025; mem0 ReAct tutorial 2024-2025
24. [InMind / InMind-Avalon — Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles](#24-inmind--inmind-avalon--evaluating-llms-in-capturing-and-applying-individual-human-reasoning-styles) — Year: 2025 (arXiv August 2025; published at EMNLP 2025 main conference)
25. [Memory & agent-architecture references — Generative Agents, ReAct / Reflexion, AgentArch, MemoryAgentBench](#25-memory--agent-architecture-references--generative-agents-react--reflexion-agentarch-memoryagentbench) — Year: ReAct 2022; Reflexion 2023; Generative Agents 2023 (UIST'23); AgentArch 2025 (arXiv Sep 2025); MemoryAgentBench 2025 (arXiv Jul 2025, ICLR 2026)
26. [Multi-agent game-environment libraries — OpenSpiel, PettingZoo, ChatArena, TextArena](#26-multi-agent-game-environment-libraries--openspiel-pettingzoo-chatarena-textarena) — Year: OpenSpiel 2019; PettingZoo 2020; ChatArena 2023 (deprecated Aug 2025); TextArena 2025
27. [TextGrad — Automatic "Differentiation" via Text](#27-textgrad--automatic-differentiation-via-text) — Year: 2024 (arXiv June 2024; published in Nature 2025)
28. [The Stackelberg Speaker — Optimizing Persuasive Communication in Social Deduction Games (also titled 'Leading the Follower: Learning Persuasive Agents in Social Deduction Games')](#28-the-stackelberg-speaker--optimizing-persuasive-communication-in-social-deduction-games-also-titled-leading-the-follower-learning-persuasive-agents-in-social-deduction-games) — Year: 2025 (arXiv October 2025; revised April 2026; ICLR 2026)

---

## Detailed Findings

## 1. Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf (Xu et al., 2023)

### Basic Info

**Name:**

> Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf (Xu et al., 2023)

- **Year:** 2023

**Authors Org:**

> Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, Yang Liu (Tsinghua University; Institute for AI Industry Research (AIR), Tsinghua University)

- **Paper Url:** https://arxiv.org/abs/2309.04658
- **Repo Url:** https://github.com/xuyuzhuang11/Werewolf

### Game Coverage

**Games Supported:**

> Werewolf (a single social-deduction game; the work positions Werewolf as a representative communication game).

- **Num Players:** 7 players with 5 role types: 2 Werewolves, 2 Villagers, 1 Witch, 1 Guard, 1 Seer.

**Hidden Role Mechanic:**

> Standard hidden-role Werewolf: Werewolves know each other and the rest of the players' roles are hidden; special roles (Seer, Witch, Guard) hold private abilities and information. Asymmetric information is conveyed through role-specific private prompts at game start.

**Rule Variant Coverage:**

> Single fixed 7-player rule configuration; no systematic coverage of multiple role sets or rule variants. Generalization is probed only by swapping the word 'werewolf' with semantically opposite terms.

- **Multimodal Support:** No. Text-only; communication and reasoning occur entirely in natural language.

### Agent Architecture

**Reasoning Paradigm:**

> Tuning-free prompting framework: frozen LLMs reason in natural language, augmented by retrieval of relevant past messages and a reflection (question-answering) step that distills observations before acting. No CoT-specific or RL/MCTS reasoning module.

**Memory Mechanism:**

> Short-term memory M accumulates observations and reflections during a single game. The framework retrieves a relevant subset of historical messages and generates reflections by self-asking and answering questions over them. Cross-game learning uses an experience pool scored by game outcome (winners scored 1000 minus game length, losers scored by game length); high-score versus median-score experiences are compared to extract actionable suggestions.

**Planning Approach:**

> No explicit search or planning module. Action choice is produced directly by the LLM conditioned on retrieved memory, reflections and extracted suggestions; planning is implicit within the prompted response.

**Theory Of Mind Evaluation:**

> No formal ToM metric. The paper qualitatively observes emergent ToM-like behavior (context-dependent trust decisions rather than blind compliance, reasoning about others' likely roles) but does not quantitatively measure belief or intent modeling accuracy.


### Game Loop Design

**Turn Structure:**

> Day/night cycle. During the day, players speak in sequence and then vote; at night, Werewolves and special roles take private actions. Turns are synchronous and host-orchestrated.

**Phase Handling:**

> A rule-based game host (built on ChatArena) manages distinct phases: night actions (Werewolf kill, Seer check, Witch save/poison, Guard protect), day discussion, and day voting; the host enforces phase transitions and win conditions.

**Communication Protocol:**

> Public natural-language chat during day discussion plus private channels for night actions and Werewolf coordination; voting is a separate structured action. Messages are routed by the host based on role visibility.

**Communication Synchrony:**

> Synchronous, turn-based discussion: players speak in a fixed sequential order each day round; there is no bidding or scheduler mechanism for deciding who speaks or when.


### Evaluation

**Metrics:**

> Winning rate by faction (Werewolf vs Villager side), average game duration in rounds, human-judged reasonableness of a sample of 50 responses, and frequency of emergent strategic behaviors (trust, confrontation, camouflage, leadership).

**Deception Metric Design:**

> No dedicated quantitative deception metric. Deception is assessed qualitatively by counting occurrences of camouflage behaviors (false role claims, fabricated events) in transcripts; there is no detector-accuracy surrogate or formal lie taxonomy.

**Persuasion Modeling:**

> Persuasive influence is not explicitly modeled or optimized. Leadership/persuasion appears only as one of the qualitatively observed emergent behaviors.

- **Rating System:** None. No Elo, TrueSkill or competitive rating; results are reported as aggregate win rates.

**Competition Format:**

> Self-play among instances of the same LLM (all seats filled by GPT-3.5-turbo agents); no cross-model tournament.

- **Llms Evaluated:** GPT-3.5-turbo-0301 (OpenAI API). Only this single model was used for the agents.

**Human Baseline Comparison:**

> No human players participate and no human win-rate baseline is provided; humans are used only to judge the reasonableness of sampled agent responses.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay generated by running the framework; experience pools are built from prior self-play games.


### Training Methodology

**Training Paradigm:**

> Tuning-free: LLM parameters are frozen. 'Learning' is achieved through retrieval, in-game reflection and a cross-game experience pool with extracted suggestions, with no gradient updates.

**Agent Controllability:**

> Not a focus. Agent strength is influenced indirectly by the size of the experience pool (10-20 historical rounds help; 30-40 become unstable), but there is no mechanism to tune to a target win rate.

**Strategy Space Analysis:**

> Yes, qualitatively. The paper analyzes emergent strategies and identifies four spontaneously arising behaviors: trust, confrontation, camouflage and leadership; it argues these reflect learned strategic patterns rather than role-name artifacts.


### Optimization

- **Prompt Optimizer Used:** None. No DSPy or automated prompt optimizer; prompts are hand-written.
- **Rl Algorithm:** None. No RL or preference optimization is used.

**Reward Design:**

> No explicit training reward. A heuristic experience-pool scoring scheme (1000 minus game length for winners, game length for losers) is used only to rank past games for suggestion extraction, not as a gradient signal.

**Optimization Target:**

> Effectively the in-context guidance: retrieved demonstrations/reflections and extracted suggestions that shape the prompt; no weights, instructions or decision chains are optimized algorithmically.

**Self Play Loop:**

> Iterative self-play generates the experience pool that later games draw on, but there is no population-based training or weight update loop.

**Credit Assignment:**

> Coarse, game-level. Credit is assigned via the outcome-based experience score over a whole game; there is no per-turn or per-utterance credit assignment.


### Engineering

**Framework Used:**

> Custom tuning-free agent framework built on ChatArena for connecting multiple LLM agents, with a rule-based game host.

- **Open Source:** Yes. Code is released on GitHub (xuyuzhuang11/Werewolf).

**Reproducibility:**

> Moderate. Code and prompts are public and the model (GPT-3.5-turbo-0301) is named, but results depend on a closed API model and self-play stochasticity; no fixed seeds/datasets guarantee exact reproduction.


### Findings

**Key Results:**

> A tuning-free, frozen-LLM framework can play 7-player Werewolf competently. Retrieval plus reflection plus an experience pool improves villager-side win rate when using a moderate amount of historical experience (10-20 rounds), while large experience pools (30-40 rounds) become unstable; moderate experience also shortens game duration. Most notably, strategic behaviors (trust, confrontation, camouflage, leadership) emerge spontaneously without being explicitly programmed.

**Identified Limitations:**

> Hallucinations degrade reasoning and factuality; the mechanism for leveraging human-derived experience is limited; no human performance baseline; cross-game generalization of learned patterns is not well controlled; experience-pool benefits are unstable at larger sizes.

**Deception Emergence:**

> Yes. Identity camouflage emerged spontaneously: agents made false role claims and fabricated events to disguise Werewolf identity. Confrontation and leadership behaviors also emerged. The authors show these persist when role names are swapped for semantically opposite terms, arguing they are learned strategic patterns rather than surface associations.

**Deceiver Detector Asymmetry:**

> Not explicitly studied or quantified. The paper notes both deceptive (camouflage) and detective (trust/confrontation) behaviors emerge but does not compare LLM strength as deceiver versus detector.


### Uncertain Fields

- Action Tool Design
- Belief State Representation
- Optimization Cost

---

## 2. Hoodwinked: Deception and Cooperation in a Text-Based Game for Language Models

### Basic Info

- **Name:** Hoodwinked: Deception and Cooperation in a Text-Based Game for Language Models
- **Year:** 2023
- **Authors Org:** Aidan O'Gara (University of Southern California / Center for AI Safety)
- **Paper Url:** https://arxiv.org/abs/2308.01404
- **Repo Url:** https://github.com/aogara-ds/hoodwinked

### Game Coverage

**Games Supported:**

> A single custom text-based social-deduction game called Hoodwinked, inspired by Mafia and Among Us. Players are locked in a house; innocents must find a key and escape while one killer eliminates them. It does not cover other named games (Werewolf, Avalon, etc.).

- **Num Players:** 4 players (1 killer/impostor + 3 innocents) in the reported experiments

**Hidden Role Mechanic:**

> One player is secretly assigned the killer role; the other players are innocents who do not know the killer's identity. Asymmetric information arises because only the killer knows the truth, and eyewitnesses may privately observe a murder while non-witnesses must infer the killer from discussion.

- **Multimodal Support:** No. Text-only game; no video, audio, or voice-prosody modalities are handled.

### Agent Architecture

**Reasoning Paradigm:**

> Direct LLM prompting (turn-by-turn response generation). No explicit ReAct, chain-of-thought, MCTS, or RL policy module; agents sample actions from enumerated choices and generate free-text discussion statements.

**Memory Mechanism:**

> Memory is supplied through personalized natural-language prompts. Each prompt to a player concatenates the rules, the player's identity/role, their action history, witnessed murders, accumulated discussion text, and prior voting results. There is no external vector store or learned memory; history is appended verbatim into the context window.

**Belief State Representation:**

> No explicit probabilistic belief table or latent belief state. Beliefs about the hidden killer are represented implicitly in natural language within the prompt context and are expressed only through votes and discussion utterances.

**Planning Approach:**

> No explicit planning module. The model generates each action or discussion statement reactively from the current game-state prompt without lookahead search or written plans.

**Action Tool Design:**

> Actions are exposed as enumerated numbered option lists embedded in the prompt (e.g., move to a room, search a location, escape; the killer additionally gets murder options targeting same-room players). The model samples from a probability distribution over these enumerated choices rather than calling external tool functions.

**Theory Of Mind Evaluation:**

> Theory of mind is evaluated indirectly through deception and detection outcomes. The paper measures whether eyewitnesses are persuaded to change votes and whether non-witnesses can infer the killer from discussion, which serves as a behavioral proxy for modeling other agents' beliefs; there is no dedicated ToM probe.


### Game Loop Design

**Turn Structure:**

> Two-stage loop. Stage 1: each player takes individual turns selecting actions (move/search rooms, escape, or for the killer, murder). Stage 2: after a murder, surviving players take turns producing discussion statements, then cast simultaneous votes to banish one player. The loop repeats until innocents escape or are all killed.

**Phase Handling:**

> Distinct action phase and discussion+voting phase. The action phase is broken by a murder event, which triggers the discussion phase followed by a banishment vote; the game then returns to the action phase.

**Communication Protocol:**

> Public natural-language discussion in a shared transcript that all surviving players see, followed by simultaneous (secret) votes to banish a player. No private channels between players.

**Communication Synchrony:**

> Synchronous turn-based. Players speak in sequence during the discussion phase, and votes are cast simultaneously after discussion concludes; there is no asynchronous bidding or scheduler-driven speech.


### Evaluation

**Metrics:**

> Killer win rate / escape rate, killer banishment rate (fraction of games where the killer is voted out), vote accuracy of witnesses vs non-witnesses, and pairwise model-vs-model comparisons of killer effectiveness.

**Deception Metric Design:**

> Deception is quantified via vote-swing surrogates: the change in the probability that a player correctly votes for the killer attributable to the discussion phase. Eyewitnesses became ~12 percentage points less likely to vote correctly after the killer's discussion (evidence the killer's lies work), while non-witnesses became ~22 percentage points more likely to identify the killer (evidence discussion aids detection). No formal lie taxonomy.

**Persuasion Modeling:**

> Persuasion is explicitly analyzed: the paper attributes stronger killers' success to superior persuasive skill in discussion rather than different in-game actions, measured by the discussion-induced shift in others' votes.

**Rating System:**

> No formal rating system (no Elo/TrueSkill). Models are compared via head-to-head pairwise win/banishment comparisons.

**Competition Format:**

> Cross-play / pairwise comparison: different LLMs are placed in the killer or innocent roles and compared across many games (24 pairwise model comparisons reported).

- **Llms Evaluated:** GPT-3 Ada, GPT-3 Curie, GPT-3.5 (chat), and GPT-4.

**Human Baseline Comparison:**

> No human baseline in the paper. The game is released publicly and the authors state human gameplay data collection is planned for future work, but no human-vs-LLM comparison is reported.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay. Results are generated entirely from self-play / cross-play of LLM agents (hundreds of games), with no human gameplay data used.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. Models are accessed via the OpenAI API with no fine-tuning, SFT, or RL; discussion statements are sampled at temperature 1 with a ~50-token cap.

**Strategy Space Analysis:**

> The paper qualitatively analyzes emergent strategies (the killer denying the crime, accusing innocents, deflecting blame) but does not formally analyze strategy-space expansion.


### Optimization

**Prompt Optimizer Used:**

> None. No DSPy or prompt-optimization technique (GEPA, MIPROv2, BootstrapFewShot, SIMBA) is used; prompts are hand-written templates.

- **Rl Algorithm:** None. No RL or preference-optimization algorithm is applied.

**Reward Design:**

> No training reward. The game outcome (escape vs. killed; killer banished vs. not) is used only as an evaluation signal, not as a training reward.

- **Optimization Target:** Nothing is optimized; the study is an evaluation of pretrained models as-is.

**Self Play Loop:**

> Self-play / cross-play is used to generate evaluation data, but there is no iterative training loop that updates the agents.

**Credit Assignment:**

> Not applicable; no learning occurs. Discussion influence is measured statistically via vote-swing rather than assigned as training credit.


### Engineering

**Framework Used:**

> Custom Python game engine that manages backend game state and procedurally generates personalized prompts; uses the OpenAI API for inference. No DSPy or LangChain.

- **Open Source:** Yes. Source code is publicly released on GitHub (github.com/aogara-ds/hoodwinked).

**Reproducibility:**

> Moderate. Code is open-sourced and the game rules and prompt construction are documented in the paper, but results depend on proprietary OpenAI API models whose versions change over time, limiting exact reproducibility.


### Findings

**Key Results:**

> More capable models are more effective killers, outperforming weaker models in 18 of 24 pairwise comparisons. The improvement is mediated by stronger persuasive skill in discussion rather than different in-game actions. Discussion increases the killer's banishment rate (e.g., 33% to 55% for GPT-3 Curie; 32% to 43% for GPT-3.5), showing discussion helps innocents overall. Advanced models were banished in ~36% of games versus ~51% for weaker models. Killers' lies measurably reduce eyewitness vote accuracy (~-12 pp) while discussion improves non-witness accuracy (~+22 pp).

**Identified Limitations:**

> Model capability does not correlate perfectly with killer effectiveness (GPT-3.5 sometimes beats GPT-4), suggesting confounders. The game is small (4 players, simple house) and only OpenAI models were tested. No human baseline is included, and the deception measure is an indirect vote-swing proxy.

**Deception Emergence:**

> Clear emergent deception: without being instructed to lie, killers spontaneously deny their crime, fabricate alibis, and accuse innocent players. These lies have a measurable causal effect, reducing eyewitnesses' probability of voting for the true killer by roughly 12 percentage points.


### Uncertain Fields

- Agent Controllability
- Deceiver Detector Asymmetry
- Optimization Cost
- Rule Variant Coverage

---

## 3. Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game (Microscopic Analysis on LLM Players via Social Deduction Game)

### Basic Info

**Name:**

> Fine-Grained and Thematic Evaluation of LLMs in Social Deduction Game (Microscopic Analysis on LLM Players via Social Deduction Game)

- **Year:** 2024
- **Paper Url:** https://arxiv.org/abs/2408.09946
- **Repo Url:** https://github.com/elu-lab/spygame

### Game Coverage

**Games Supported:**

> Spyfall (a social deduction game where one or more 'spies' do not know a shared secret location and must blend in via question-and-answer rounds)

**Hidden Role Mechanic:**

> Asymmetric information via a hidden spy role: non-spy players share a common location/topic while the spy is uninformed and must infer it; spies attempt to evade detection while non-spies attempt to expose the spy through indirect questioning

- **Multimodal Support:** No; purely text-based evaluation with no video, audio, or voice-prosody modalities

### Agent Architecture

**Reasoning Paradigm:**

> Prompted LLM agents performing question-asking, answering, accusation and deduction in natural language; no specialized search or RL policy is used (the contribution is evaluation methodology, not a new reasoning architecture)


### Game Loop Design

**Communication Protocol:**

> Public natural-language question-and-answer exchanges among all players, followed by a public accusation; no private channels


### Evaluation

**Persuasion Modeling:**

> Persuasive influence is not explicitly modeled or optimized; the focus is on diagnostic fine-grained skill evaluation

- **Competition Format:** LLM-vs-LLM self-play game sessions used as a measurement harness; no cross-play tournament

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay generated by running the Spyfall environment, analyzed with quantitative metrics and qualitative thematic coding


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting; no SFT, RL, or DPO is applied. The contribution is an evaluation framework, not a trained agent

- **Agent Controllability:** Not applicable; agent skill is not tuned to a target win rate or difficulty

**Strategy Space Analysis:**

> Emergent failure modes are analyzed thematically (four reasoning-failure categories) rather than emergent winning strategies or strategy-space expansion


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method (GEPA, MIPROv2, BootstrapFewShot, SIMBA) is used
- **Rl Algorithm:** None; no RL or preference-optimization algorithm is used
- **Reward Design:** Not applicable; no training signal or reward is defined since agents are not trained
- **Optimization Target:** Not applicable; nothing is optimized (evaluation-only study)

**Self Play Loop:**

> LLM-vs-LLM self-play is used only to generate gameplay data for evaluation, not for iterative training

**Credit Assignment:**

> Not applicable; no credit assignment since no learning occurs. The methodological contribution instead replaces coarse outcome credit with event-level fine-grained metrics


### Engineering

- **Open Source:** Yes; code is released at github.com/elu-lab/spygame

### Uncertain Fields

- Action Tool Design
- Authors Org
- Belief State Representation
- Communication Synchrony
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Framework Used
- Human Baseline Comparison
- Identified Limitations
- Key Results
- Llms Evaluated
- Memory Mechanism
- Metrics
- Num Players
- Optimization Cost
- Phase Handling
- Planning Approach
- Rating System
- Reproducibility
- Rule Variant Coverage
- Theory Of Mind Evaluation
- Turn Structure

---

## 4. Learning to Discuss Strategically: A Case Study on One Night Ultimate Werewolf

### Basic Info

- **Name:** Learning to Discuss Strategically: A Case Study on One Night Ultimate Werewolf
- **Year:** 2024

**Authors Org:**

> Xuanfa Jin, Ziyan Wang, Yali Du, Meng Fang, Haifeng Zhang, Jun Wang — Institute of Automation, Chinese Academy of Sciences; King's College London; University of Liverpool; University College London. Published at NeurIPS 2024

- **Paper Url:** https://arxiv.org/abs/2405.19946 (NeurIPS 2024; https://openreview.net/forum?id=1f82rnwCbl)
- **Repo Url:** https://github.com/KylJin/Werewolf

### Game Coverage

**Games Supported:**

> One Night Ultimate Werewolf (ONUW), a variant of Werewolf with night role changes; a simplified version with two Werewolves and one Robber is analyzed theoretically, and five-player configurations are used for experiments

**Num Players:**

> 5 players in the experimental setting (e.g., roles including Werewolf, Seer, Robber, Troublemaker, Insomniac); a simplified 3-role version (2 Werewolves + 1 Robber) is used for the game-theoretic analysis

**Hidden Role Mechanic:**

> Hidden roles with role-changing night actions: roles can be swapped during the single night phase (e.g., the Robber steals another player's role), so a player's true role is uncertain even to themselves, sharply increasing information asymmetry and complexity over standard Werewolf

**Rule Variant Coverage:**

> Single game (ONUW); two analysis scenarios (discussion vs. no-discussion) and two difficulty levels (easy/hard) of the same game, but no broader role-set variant coverage

- **Multimodal Support:** No; text-based ONUW only, with no video, audio, or voice-prosody modalities

### Agent Architecture

**Reasoning Paradigm:**

> RL-instructed LLM agent: an RL-trained discussion policy selects a high-level discussion tactic, which then conditions LLM-based belief modeling and decision making; the game is formalized as a Multi-Phase Extensive-Form Bayesian Game (MP-EFBG)

**Belief State Representation:**

> Explicit belief modeling: the agent forms natural-language beliefs about each player's role based on observations, and these beliefs (significantly shaped by discussion) drive voting decisions; discussion tactics are treated as latent variables conditioning behavioral strategies

**Action Tool Design:**

> Actions are decomposed by game phase: night role actions, day discussion utterances, and voting; the RL policy chooses among a discrete candidate set of discussion tactics, and the LLM then realizes the chosen tactic as natural-language speech and makes phase-appropriate decisions

**Theory Of Mind Evaluation:**

> ToM is handled via explicit belief modeling — the agent reasons about other players' roles — and the paper analyzes how discussion changes players' beliefs and utilities; there is no separate quantitative ToM-accuracy metric, but average votes received is used as a role-concealment indicator


### Game Loop Design

**Turn Structure:**

> Four phases per game: role assignment, night actions, day discussion (three rounds), and voting; within discussion, players take sequential speaking turns

**Phase Handling:**

> Distinct phases — role assignment, night actions, three-round day discussion, and voting — are managed by the ONUW environment, with the agent selecting phase-appropriate behavior

**Communication Protocol:**

> Public natural-language discussion across three day rounds plus public voting; secret night role actions are private


### Evaluation

**Metrics:**

> Win rate across team alignments; NashConv values measuring how closely strategies approximate the Perfect Bayesian Equilibrium; and average number of votes received as an indicator of role concealment

**Deception Metric Design:**

> Deception/concealment is quantified through average votes received (lower means better role concealment / more successful evasion) and through the choice of deceptive discussion tactics; no explicit lie taxonomy or detector-accuracy surrogate

**Persuasion Modeling:**

> Persuasion is explicitly central: the paper shows discussion greatly changes players' beliefs and utilities, and the RL-trained discussion policy is optimized to select tactics (including persuasive/deceptive ones) that influence others' beliefs

**Rating System:**

> No Elo/TrueSkill; head-to-head win-rate comparisons across team alignments, plus NashConv as an equilibrium-distance measure


### Training Methodology

**Training Paradigm:**

> RL-instructed: an offline reinforcement learning discussion policy is trained on synthetic game data while the LLMs themselves remain tuning-free (used as fixed backends conditioned by the learned policy)

**Agent Controllability:**

> Not a focus; the agent is optimized for strategic discussion performance, not tuned to a target win rate or difficulty

**Strategy Space Analysis:**

> The work analyzes how discussion tactics affect equilibria and beliefs, and demonstrates the existence of Perfect Bayesian Equilibria with and without discussion, but does not perform iterative strategy-space expansion


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method is used

**Optimization Target:**

> A discussion policy (RL policy weights) that selects high-level discussion tactics; the LLM backends are not fine-tuned


### Engineering

- **Open Source:** Yes; the ONUW environment and code are released at github.com/KylJin/Werewolf

### Findings

**Key Results:**

> The paper formalizes ONUW as a Multi-Phase Extensive-Form Bayesian Game and proves the existence of Perfect Bayesian Equilibria in both discussion and no-discussion scenarios, showing discussion substantially changes players' beliefs and utilities. The proposed RL-instructed agent, which uses an RL-trained discussion policy plus belief modeling, outperforms baseline agents: it achieves a 0.70 win rate with a Gemini backend and 0.50 with a GPT-4 backend, while maintaining the lowest average votes received (1.10), indicating the strongest role concealment. Agents trained with the framework also more closely approximate the theoretical equilibria (better NashConv)

**Deception Emergence:**

> Emergent strategic deception is demonstrated: the RL-instructed agent learns to choose discussion tactics — including deceptive accusations and selective role concealment — that minimize suspicion (lowest average votes received), showing it actively manages others' beliefs rather than speaking honestly


### Uncertain Fields

- Communication Synchrony
- Competition Format
- Credit Assignment
- Data Source Provenance
- Deceiver Detector Asymmetry
- Framework Used
- Human Baseline Comparison
- Identified Limitations
- Llms Evaluated
- Memory Mechanism
- Optimization Cost
- Planning Approach
- Reproducibility
- Reward Design
- Rl Algorithm
- Self Play Loop

---

## 5. Murder-Mystery Benchmarks: WhodunitBench (NeurIPS 2024) and ThinkThrice / 'Deciphering Digital Detectives' Jubensha (ACL 2024 Findings)

### Basic Info

**Name:**

> Murder-Mystery Benchmarks: WhodunitBench (NeurIPS 2024) and ThinkThrice / 'Deciphering Digital Detectives' Jubensha (ACL 2024 Findings)

- **Year:** 2024

**Authors Org:**

> WhodunitBench: Junlin Wang et al. (incl. Guanbin Li; Sun Yat-sen University and collaborators), NeurIPS 2024 Datasets & Benchmarks (spotlight). ThinkThrice: Dekun Wu, Haochen Shi, Zhiyuan Sun, Bang Liu et al. (Universite de Montreal / Mila), ACL 2024 Findings.

**Paper Url:**

> WhodunitBench: https://openreview.net/forum?id=qmvtDIfbmS ; ThinkThrice/Jubensha: https://arxiv.org/abs/2312.00746

**Repo Url:**

> WhodunitBench: https://github.com/jun0wanan/WhodunitBench-Murder_Mystery_Games ; ThinkThrice: https://github.com/jackwu502/ThinkThrice


### Game Coverage

**Games Supported:**

> Both target murder-mystery / script-murder (Jubensha, 'Scripted Murders', 剧本杀) deduction games. WhodunitBench uses 50 curated multimodal murder-mystery scripts; ThinkThrice/Jubensha uses Chinese Jubensha scripts where players read character scripts to deduce the murderer. Neither covers Werewolf/Mafia/Avalon.

**Hidden Role Mechanic:**

> Each player receives a private character script invisible to others; one player is the murderer. The murderer knows the truth and may lie strategically, while other characters hold partial private information and must honestly share clues to collectively identify the killer. Asymmetric information is distributed across per-character scripts.

**Rule Variant Coverage:**

> WhodunitBench spans 50 distinct murder-mystery scripts; ThinkThrice draws on a corpus of 1,115 Jubensha scripts (4-20 players, 4k-518k tokens each). Breadth is in scenario/script diversity rather than distinct rule systems; the core deduction rules are fixed.

**Multimodal Support:**

> WhodunitBench is explicitly multimodal: it evaluates large multimodal agents using image clues plus text, requiring multimodal perception. The Jubensha dataset includes text/audio/video material, but the ThinkThrice experiments are text-only.


### Agent Architecture

**Reasoning Paradigm:**

> WhodunitBench evaluates multimodal perception, interaction, reasoning, and decision-making by LMAs. ThinkThrice introduces a three-stage reasoning pipeline (Memory Retrieval, Self-Refinement, Self-Verification) layered on standard LLM prompting; no RL or MCTS.

**Belief State Representation:**

> Beliefs are represented in natural language. Agents accumulate case knowledge from dialogue and scripts and express deductions in text; there is no explicit probabilistic belief table. WhodunitBench probes belief reasoning through multiple-choice and open-ended questions rather than a maintained belief state.

**Planning Approach:**

> No explicit search-based planning. ThinkThrice agents generate responses and questions based on current dialogue, retrieved memories, character relations, and rules, with Self-Refinement decomposing questions into sub-questions. WhodunitBench evaluates multi-step decision-making but agents do not run formal planning algorithms.

**Action Tool Design:**

> Actions are exposed as natural-language generation orchestrated by a host/game controller: agents answer questions, ask questions of other players, examine clues, and cast votes. A host module issues pre-set commands to drive game progression. Actions are not external callable functions.

**Theory Of Mind Evaluation:**

> Both highlight theory of mind. WhodunitBench explicitly concludes that fully applying theory of mind to complete a game like a human remains a significant challenge. ThinkThrice measures cross-character inference (integrating information from others' scripts) via inferential QA, an indirect ToM probe; neither maintains an explicit ToM model.


### Game Loop Design

**Turn Structure:**

> Synchronous turn-taking: one agent acts at a time. ThinkThrice runs an 8-stage game flow (script distribution, self-introduction, initial questioning, two open-questioning rounds, clue distribution, three final-questioning rounds, anonymous voting). WhodunitBench runs an arena-style full-game playthrough plus a staged 'chain of evaluation'.

**Phase Handling:**

> Distinct phases are managed by a host/game controller. ThinkThrice sequences introduction, multiple questioning rounds, clue release, and a final voting phase. WhodunitBench separates an arena game-play mode from a multi-stage QA evaluation chain.

**Communication Protocol:**

> Public natural-language dialogue: agents introduce themselves, question one another, and respond, followed by anonymous voting. Private information stays in each agent's character script; there is no private chat channel.

**Communication Synchrony:**

> Synchronous and turn-based: agents respond to the host and to other players sequentially; only one agent speaks at a given moment, with no asynchronous bidding or scheduler-driven speech.


### Evaluation

**Metrics:**

> ThinkThrice: factual QA accuracy (own vs others' scripts), chat-history-to-script similarity (embeddings/TF-IDF/trigram), inferential QA Overall Accuracy and Informed Accuracy (answer + valid reasoning), civilian win rate, and murderer-identification accuracy. WhodunitBench: scores over 3000+ multiple-choice and open-ended questions plus arena win/identification outcomes across four ability areas.

**Persuasion Modeling:**

> Not modeled. Both focus on information gathering and multi-step reasoning; persuasive influence and rhetorical strategy are not explicitly measured or optimized.

**Rating System:**

> No competitive rating system (no Elo/TrueSkill). Performance is reported as accuracy scores and win/identification rates.

**Competition Format:**

> Both are evaluation benchmarks rather than tournaments. Agents (single model controlling all roles, or LMAs under test) play through scripted games; WhodunitBench adds an arena-style playthrough plus a structured QA chain.

**Human Baseline Comparison:**

> No human-player baseline in either benchmark. ThinkThrice uses two native-speaker annotators only to validate automatic metrics on subsamples (~200 factual and ~100 inferential QA items), with 0.7-0.95 correlation to automatic scores. WhodunitBench compares LMA performance against human-level expectations qualitatively.

**Data Source Provenance:**

> Curated real human-authored game material played by LLM/LMA agents. ThinkThrice collected 1,115 real Jubensha scripts from online platforms; WhodunitBench curated 50 murder-mystery scripts with image and text clues. Gameplay itself is synthetic LLM-vs-LLM.


### Training Methodology

**Training Paradigm:**

> Tuning-free. No agent training; both rely on in-context learning and prompt engineering. ThinkThrice adds inference-time Self-Refinement and Self-Verification loops that regenerate answers until authenticity thresholds are met.

**Strategy Space Analysis:**

> Limited. WhodunitBench analyzes capability gaps in multi-agent collaboration and multi-step reasoning; ThinkThrice analyzes which pipeline components help, but neither performs a formal emergent-strategy / strategy-space-expansion analysis.


### Optimization

**Prompt Optimizer Used:**

> None. No DSPy or prompt optimizer (GEPA, MIPROv2, BootstrapFewShot, SIMBA); prompts are hand-engineered.

- **Rl Algorithm:** None. No RL or preference-optimization algorithm is used.

**Reward Design:**

> No training reward. Game outcomes and QA accuracy are used purely as evaluation signals. ThinkThrice's Self-Verification uses an internal authenticity-threshold check to gate answer acceptance, not a learning reward.

**Optimization Target:**

> Nothing is trained; the studies evaluate pretrained models. ThinkThrice's pipeline optimizes answer quality at inference time via refinement/verification rather than optimizing weights or prompts.

- **Self Play Loop:** No iterative self-play training loop. Games are run for evaluation only.

**Credit Assignment:**

> Not applicable; no learning. ThinkThrice attributes answer quality to specific pipeline modules via ablations rather than multi-turn credit assignment.


### Engineering

**Framework Used:**

> Custom Python implementations. ThinkThrice uses Faiss for vector memory plus OpenAI APIs and structured prompting; WhodunitBench provides a custom evaluation engine (e.g., GameStart.py) with curated multimodal datasets. Neither uses DSPy or LangChain.

**Open Source:**

> Yes. Both release code and data: WhodunitBench on github.com/jun0wanan/WhodunitBench-Murder_Mystery_Games and ThinkThrice on github.com/jackwu502/ThinkThrice.

**Reproducibility:**

> Moderate. Both release code and datasets and document their pipelines; however results depend on proprietary OpenAI/multimodal models whose versions change, and LLM stochasticity causes variance (ThinkThrice mitigates with 3 runs per game).


### Findings

**Key Results:**

> WhodunitBench: current large multimodal agents do acceptably on basic perceptual tasks but are insufficiently equipped for complex multi-agent collaboration and multi-step reasoning, and full human-like theory-of-mind use to complete a game remains a significant challenge. ThinkThrice: the full Memory-Retrieval + Self-Refinement + Self-Verification pipeline raises performance substantially (e.g., GPT-4 with full-script access reaches ~60% overall inferential accuracy with the pipeline vs ~15% without); agents answer questions about their own script far better (0.772) than about others' scripts (0.495), and reach 0.831 similarity to full scripts, showing weakness at integrating others' hidden information.

**Identified Limitations:**

> WhodunitBench: agents struggle with multi-agent collaboration, long multi-step reasoning, and theory of mind. ThinkThrice: Chinese-only dataset limits cross-linguistic generalization; high LLM stochasticity; reproducibility complicated by OpenAI model-version drift; substantial per-game cost; agents weak at reasoning over other characters' hidden information.


### Uncertain Fields

- Agent Controllability
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Llms Evaluated
- Memory Mechanism
- Num Players
- Optimization Cost

---

## 6. Strategic-Game Meta-Benchmarks: DSGBench, GTBench, GAMA-Bench, AgentBench

### Basic Info

- **Name:** Strategic-Game Meta-Benchmarks: DSGBench, GTBench, GAMA-Bench, AgentBench
- **Year:** 2024

**Authors Org:**

> DSGBench: Wenjie Tang, Yuan Zhou, Erqiang Xu, Keyan Cheng, Minne Li, Liquan Xiao (NUDT / DeciBrain-Group), 2025. GTBench: Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, Kaidi Xu (Drexel, BU, LLNL, Lehigh, UNC/MIT), NeurIPS 2024. GAMA-Bench: Jen-tse Huang et al. (CUHK ARISE), 2024. AgentBench: Xiao Liu et al. (Tsinghua / THUDM and collaborators), ICLR 2024.

**Paper Url:**

> DSGBench: https://arxiv.org/abs/2503.06047 ; GTBench: https://arxiv.org/abs/2402.12348 ; GAMA-Bench: https://arxiv.org/abs/2403.11807 ; AgentBench: https://arxiv.org/abs/2308.03688

**Repo Url:**

> DSGBench: https://github.com/DeciBrain-Group/DSGBench ; GTBench: https://github.com/jinhaoduan/GTBench ; GAMA-Bench: https://github.com/CUHK-ARISE/GAMABench ; AgentBench: https://github.com/THUDM/AgentBench


### Game Coverage

**Games Supported:**

> These are multi-game strategic-reasoning meta-benchmarks; only DSGBench directly includes a social-deduction game. DSGBench: 6 games (StarCraft II, Civilization, Street Fighter III, Diplomacy, Werewolf, Stratego) - Werewolf is the social-deduction game and Diplomacy is negotiation-heavy. GTBench: 10 game-theoretic games (Tic-Tac-Toe, Connect-4, Breakthrough, Kuhn Poker, Liar's Dice, Nim, Negotiation, Pig, Iterated Prisoner's Dilemma, Blind Auction). GAMA-Bench: 8 classic game-theory scenarios (Guess 2/3 of the Average, El Farol Bar, Divide the Dollar, Public Goods Game, Diner's Dilemma, Sealed-Bid Auction, Battle Royale, Pirate Game). AgentBench: 8 agent environments (OS, DB, Knowledge Graph, Digital Card Game/Aquawar, Lateral Thinking Puzzles, House-Holding/ALFWorld, Web Shopping, Web Browsing) - no dedicated social-deduction game.

**Hidden Role Mechanic:**

> Only DSGBench's Werewolf uses true hidden roles (werewolves secretly assigned among villagers; deduction via discussion). GTBench's Liar's Dice / Kuhn Poker and GAMA-Bench's auctions involve hidden information but not hidden roles; AgentBench has no hidden-role mechanic.

**Rule Variant Coverage:**

> Breadth across games rather than role-set variants: DSGBench 6 games, GTBench 10 games, GAMA-Bench 8 games, AgentBench 8 environments. Within the social-deduction slice only DSGBench's single Werewolf configuration is covered; none of the four sweeps multiple social-deduction role sets.


### Agent Architecture

**Reasoning Paradigm:**

> Prompt-based LLM reasoning, evaluated as agents. GTBench explicitly compares reasoning methods (it tests Chain-of-Thought and Tree-of-Thought prompting). DSGBench, GAMA-Bench, and AgentBench use standard prompted-agent setups; none mandate ReAct, MCTS, or RL policies, though AgentBench evaluates multi-round agentic interaction.

**Planning Approach:**

> Planning is left to the agent's prompted reasoning. GTBench evaluates Tree-of-Thought as a planning-style method; DSGBench evaluates strategic planning as one scored dimension; AgentBench probes long-horizon planning across multi-step environments. No benchmark mandates explicit search-based planners.

**Action Tool Design:**

> Actions are exposed as structured text interfaces per game. GTBench is built on OpenSpiel, exposing legal actions per game state; DSGBench wraps environments with an OpenAI-Gym-style interface; AgentBench provides standardized action/observation protocols per environment; GAMA-Bench exposes game-specific decision formats. Actions are parsed from agent text rather than being external callable tools.


### Game Loop Design

**Turn Structure:**

> Game-specific. GTBench (via OpenSpiel) uses each game's native turn order; DSGBench mixes real-time (StarCraft II, Street Fighter III) and turn-based (Civilization, Diplomacy, Werewolf, Stratego) structures; GAMA-Bench uses simultaneous or sequential moves per scenario; AgentBench uses multi-turn step-by-step interaction with environments.


### Evaluation

**Metrics:**

> DSGBench: fine-grained scoring across five dimensions (strategic planning, real-time decision-making, team collaboration / multi-agent interaction, adaptability, and reasoning/cognitive ability) plus win rate and decision-trajectory analysis. GTBench: normalized relative advantage / win rates across the 10 games. GAMA-Bench: a dynamic per-game scoring scheme aggregated to a 0-100 score. AgentBench: per-environment success/score aggregated into an overall agent score.

**Persuasion Modeling:**

> Not explicitly modeled. Persuasive negotiation matters in DSGBench's Diplomacy/Werewolf and GTBench's Negotiation, but influence is reflected only in match outcomes rather than measured as a separate persuasion metric.

**Competition Format:**

> Primarily cross-play / model-vs-model and model-vs-environment evaluation. GTBench pits LLMs head-to-head (and against conventional/MCTS-style agents); DSGBench runs agents in shared multi-agent games; GAMA-Bench runs multi-agent scenarios; AgentBench evaluates each model independently against fixed environments.

**Llms Evaluated:**

> DSGBench: GPT-3.5, GPT-4o, GPT-4o-mini, o1-mini, DeepSeek-V2.5, Llama-3.1-70B, Gemini-1.5-Flash. GTBench: GPT-3.5/GPT-4, CodeLlama-34B-Instruct, Llama-2-70B-chat, Llama-3-70B-Instruct, Mixtral and other open models. GAMA-Bench: 13 models across 6 families (GPT-3.5 x3, GPT-4 x2, Gemini-1.0/1.5-Pro, Llama-3.1 8B/70B/405B, Mixtral-8x7B/8x22B, Qwen-2-72B). AgentBench: 29 API-based and open-source LLMs.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM / LLM-vs-environment gameplay generated by running the benchmarks; GTBench additionally generates LLM-vs-conventional-agent data. No curated real human gameplay datasets.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting for evaluation. All four evaluate pretrained models as-is; AgentBench discusses (but the benchmark itself does not perform) that multi-round alignment / code training affects agent ability. No SFT/RL is part of the benchmark protocols.

**Strategy Space Analysis:**

> Yes, partially. GTBench analyzes how LLM behavior varies across game categories (complete vs incomplete information, deterministic vs probabilistic) and finds CoT/ToT do not always help; DSGBench's decision-tracking mechanism analyzes strategy turning points; GAMA-Bench analyzes deviation from game-theoretic equilibria; AgentBench analyzes failure modes. None performs a formal strategy-space-expansion study.


### Optimization

**Prompt Optimizer Used:**

> None. No DSPy or prompt optimizer (GEPA, MIPROv2, BootstrapFewShot, SIMBA) is used; GTBench compares hand-designed prompting methods (CoT, ToT) but not automated optimizers.

**Rl Algorithm:**

> None. The benchmarks are evaluation suites and do not apply RL or preference-optimization algorithms.

**Reward Design:**

> No training reward. Game outcomes and per-dimension scores serve purely as evaluation signals (win rate, normalized relative advantage, dynamic scores, task success).

**Optimization Target:**

> Nothing is trained by the benchmarks; they evaluate pretrained models. GTBench additionally compares prompting strategies as the variable under study.

**Self Play Loop:**

> No iterative self-play training loop. Games / cross-play are used only to generate evaluation results.

**Credit Assignment:**

> Not applicable; no learning occurs. DSGBench's decision-tracking and GTBench's per-game breakdowns provide post-hoc attribution of behavior rather than training credit assignment.


### Engineering

**Framework Used:**

> Custom Python evaluation frameworks. GTBench is built on DeepMind OpenSpiel; DSGBench uses modular game environments with an OpenAI-Gym-style interface and Weights & Biases tracking; GAMA-Bench and AgentBench provide their own multi-game / multi-environment harnesses. None uses DSPy or LangChain as the core framework.

**Open Source:**

> Yes. All four release code and evaluation packages on GitHub (DeciBrain-Group/DSGBench, jinhaoduan/GTBench, CUHK-ARISE/GAMABench, THUDM/AgentBench).

**Reproducibility:**

> Generally good for meta-benchmarks: all provide open code, game environments, and configs. GTBench's OpenSpiel grounding aids reproducibility. Main caveats are dependence on proprietary API models with version drift and LLM stochasticity across runs.


### Findings

**Key Results:**

> GTBench: most open-source LLMs (e.g., CodeLlama-34B, Llama-2-70B-chat) are less competitive than commercial LLMs like GPT-4 in complex games, though Llama-3-70B-Instruct closes the gap; code-pretraining benefits strategic reasoning; CoT/ToT do not always help; LLMs are weak in complete/deterministic games but more competitive in probabilistic ones. GAMA-Bench: Gemini-1.5-Pro leads with 69.8/100, followed by Llama-3.1-70B (65.9) and Mixtral-8x22B (62.4); LLMs deviate from game-theoretic optima. DSGBench: distinct strengths and systemic limitations emerge across the six games, revealed via decision-trajectory analysis; agents struggle with cross-scenario generalization. AgentBench: top commercial LLMs act capably as agents but open-source models lag significantly; poor long-horizon reasoning, decision-making, and instruction-following are the main obstacles.

**Identified Limitations:**

> LLM agents show weak long-horizon reasoning and limited cross-game generalization; performance is uneven across game categories; advanced reasoning prompts (CoT/ToT) do not reliably help; open-source models trail commercial ones. Social-deduction coverage is shallow (only DSGBench's single Werewolf setup), and results depend on proprietary models and are sensitive to stochasticity.


### Uncertain Fields

- Agent Controllability
- Belief State Representation
- Communication Protocol
- Communication Synchrony
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Human Baseline Comparison
- Memory Mechanism
- Multimodal Support
- Num Players
- Optimization Cost
- Phase Handling
- Rating System
- Theory Of Mind Evaluation

---

## 7. Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction (Google, 2024)

### Basic Info

- **Name:** Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction (Google, 2024)
- **Year:** 2024
- **Authors Org:** Suma Bailis, Jane Friedhoff, Feiyang Chen (Google Research / Google)
- **Paper Url:** https://arxiv.org/abs/2407.13943
- **Repo Url:** https://github.com/google/werewolf_arena

### Game Coverage

- **Games Supported:** Werewolf (single social-deduction game) used as a case-study evaluation arena.
- **Num Players:** 8 players per game: 1 Seer, 1 Doctor, 2 Werewolves, 4 Villagers.

**Hidden Role Mechanic:**

> Standard hidden-role Werewolf: Werewolves know each other; Villagers, Seer and Doctor have hidden identities. Each agent receives role-specific privileged information; the Seer privately investigates roles and the Doctor privately protects players.

**Rule Variant Coverage:**

> Single fixed 8-player role configuration; the framework does not systematically vary role sets or rule variants.

- **Multimodal Support:** No. Text-only natural-language gameplay.

### Agent Architecture

**Reasoning Paradigm:**

> Prompt-based reasoning over a memory stream; agents summarize and reflect each round and reason about other players' intentions. No RL policy or explicit search; reasoning is LLM-internal, guided by structured prompts.

**Memory Mechanism:**

> Each agent maintains a memory stream containing observational memories (game-level events) and reflective memories, plus role-specific privileged information. At the end of every round agents perform a summarization/distillation step that compresses key insights from the debate into the memory stream for use in later rounds.

**Belief State Representation:**

> Natural-language beliefs maintained within the memory stream and reflections; agents adjust beliefs about likely roles based on debate dynamics. Belief impact is probed externally via a synthetic-voting measurement rather than an explicit probability table.

**Planning Approach:**

> No explicit search or formal plan. Agents decide bids, debate content, votes and night actions directly from prompted reasoning over the memory stream; planning is implicit in the per-round summarize-and-act loop.

**Action Tool Design:**

> Actions are structured outputs orchestrated by a rules-based Game Master: bidding (interest level), debate utterance, vote, and role-specific night actions (Werewolf eliminate, Doctor protect, Seer investigate). Actions are not exposed as free-form callable tools but as fixed prompted decision points the GM parses.

**Theory Of Mind Evaluation:**

> Partially. The framework uses synthetic voting (asking agents how they would vote after each utterance) to quantify how speech shifts others' beliefs, plus Seer believability and backfire-rate metrics; this indirectly measures modeling of and influence on other agents' beliefs.


### Game Loop Design

**Turn Structure:**

> Rounds alternate between a Night phase and a Day phase. Day debate proceeds with a bidding-determined speaking order, capped at 8 debate turns, followed by a vote.

**Phase Handling:**

> A rules-based Game Master orchestrates phases: Night (simultaneous Werewolf elimination, Doctor protection, Seer investigation), Day debate, and Day voting requiring a majority to exile a player; the GM enforces transitions and win conditions.

**Communication Protocol:**

> Public natural-language debate during the day, private channels for night actions and Werewolf coordination, and explicit majority-rule voting. Communication is mediated by the Game Master.

**Communication Synchrony:**

> Synchronous turn-based debate but with a dynamic bidding-based turn-taking system: agents bid an interest level (0 observe, 1 general thoughts, 2 critical contribution, 3 urgent, 4 direct response); the highest bidder speaks next, with ties broken in favor of recently mentioned players. This replaces fixed or random speaking order.


### Evaluation

**Metrics:**

> Win rate (Villager vs Werewolf victories), bid-distribution patterns, voting entropy (consensus formation), and Seer-specific metrics: reveals per game, reveal timing, reveal accuracy, believability, and backfire rate. Synthetic voting is used to measure per-utterance dialogue impact.

**Deception Metric Design:**

> Deception is measured indirectly rather than via a formal lie taxonomy: synthetic voting captures vote-swing caused by utterances, and Seer believability/backfire rate captures whether claims (true or false) are accepted. No detector-accuracy surrogate.

**Persuasion Modeling:**

> Yes, indirectly. Persuasive influence is operationalized through synthetic voting (how an utterance shifts others' votes) and the bidding mechanism that lets agents seize the floor to shape debate; persuasion is measured but not directly optimized.

- **Rating System:** Elo. A round-robin tournament produces an Elo leaderboard across the evaluated models.

**Competition Format:**

> Cross-play tournament: a round-robin among 7 LLMs in which each model pairing plays 10 matches in a controlled setup, then Elo is computed.

**Llms Evaluated:**

> Gemini family (Gemini 1.5 Pro, Gemini Pro, Gemini Flash) and OpenAI GPT family (GPT-4, GPT-4o, GPT-3.5).

- **Human Baseline Comparison:** No human players; evaluation is entirely LLM-vs-LLM. Humans are not used as a performance baseline.
- **Data Source Provenance:** Synthetic LLM-vs-LLM gameplay generated by the arena's tournament runs.

### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. All LLMs are used off-the-shelf with structured prompts; no fine-tuning, SFT or RL.

**Agent Controllability:**

> Not a focus; agent strength is determined by the underlying model. No mechanism to tune an agent to a target win rate or difficulty.

**Strategy Space Analysis:**

> Yes, qualitatively. The paper analyzes behavioral differences (e.g., GPT-4 favoring longer, formal utterances vs Gemini 1.5 Pro using shorter, more emotionally expressive speech) and Seer reveal strategies, but does not study strategy-space expansion.


### Optimization

- **Prompt Optimizer Used:** None. Prompts are hand-written and released; no DSPy or automated prompt optimizer.
- **Rl Algorithm:** None. No RL or preference optimization.

**Reward Design:**

> No training reward. The win/loss game outcome feeds only the Elo computation, not any optimization signal.

**Optimization Target:**

> Nothing is algorithmically optimized; the arena is an evaluation harness rather than a training method.

**Self Play Loop:**

> Cross-play tournament games are run for evaluation, but there is no iterative self-play training loop that updates agents.

**Credit Assignment:**

> Game-level only, via Elo from win/loss outcomes; synthetic voting attributes per-utterance influence for analysis but not for credit assignment in any learning sense.


### Engineering

**Framework Used:**

> Custom open-source Werewolf Arena framework with a rules-based Game Master orchestrating the game loop.

- **Open Source:** Yes. Code and all prompts are released on GitHub (google/werewolf_arena).

**Reproducibility:**

> Moderate-to-good. Code and prompts are public and models are named, but several evaluated models are closed APIs and only 10 games per pairing are run, so exact numbers are sensitive to model versions and stochasticity.


### Findings

**Key Results:**

> Werewolf Arena demonstrates social-deduction Werewolf as a viable LLM evaluation harness. A bidding-based dynamic turn-taking system lets agents strategically seize the floor (e.g., Werewolves bidding high to defend an exposed teammate). In the round-robin Elo tournament, Gemini 1.5 Pro outperformed GPT-4 overall and excelled as a Villager. Stylistic differences emerged: GPT-4 produced longer, more formal utterances while Gemini 1.5 Pro used shorter, less frequent, more emotionally expressive speech.

**Identified Limitations:**

> Simplified game environment; limited sample size (10 games per model pairing); acknowledged dual-use concerns around optimizing persuasive/deceptive language. Results are sensitive to model versions.

**Deception Emergence:**

> Yes. Werewolf agents engage in strategic defense and concealment, e.g., a Werewolf bidding urgently to defend an accused teammate while villagers stay quiet; Seer reveal timing and believability dynamics show emergent strategic information management.

**Deceiver Detector Asymmetry:**

> Not explicitly framed as a deceiver-vs-detector asymmetry study; the paper reports faction win rates and role-specific performance but does not isolate LLM strength as deceiver versus detector.


### Uncertain Fields

- Optimization Cost

---

## 8. Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies (WereBench / WereAlign)

### Basic Info

**Name:**

> Beyond Survival: Evaluating LLMs in Social Deduction Games with Human-Aligned Strategies (WereBench / WereAlign)

- **Year:** 2025

**Authors Org:**

> Zirui Song, Yuan Huang, Junchang Liu, Haozhe Luo, Chenxi Wang, Lang Gao, Zixiang Xu, Mingfei Han, Xiaojun Chang, Xiuying Chen (Mohamed bin Zayed University of Artificial Intelligence (MBZUAI); Northeastern University)

- **Paper Url:** https://arxiv.org/abs/2510.11389

### Game Coverage

**Games Supported:**

> Werewolf (social-deduction). The benchmark focuses on Werewolf as played in a professional televised setting.

**Num Players:**

> Variable player counts; the WereBench dataset covers 48 unique human players and 30 unique roles across 15 rule variants (specific per-game seat counts vary by variant).

**Hidden Role Mechanic:**

> Standard hidden-role Werewolf with 30 distinct roles across 15 rule variants. Asymmetric information is reconstructed from public game logs and day/night cycles; the benchmark tests whether models can infer hidden roles and intentions from observed play.

**Rule Variant Coverage:**

> Broad: 15 distinct rule variants and 30 unique roles, deliberately chosen to test generalization beyond a single fixed configuration.

**Multimodal Support:**

> Yes. The dataset includes 100+ hours of video with dynamic camera work capturing non-verbal cues such as microexpressions, alongside transcribed speech; it is described as a multimodal Werewolf dataset.


### Agent Architecture

**Reasoning Paradigm:**

> Evaluation-as-benchmark rather than an agent: models answer multiple-choice and decision tasks (role inference, strategic judgment, deception reasoning, persuasive statements, counterfactual trade-off) over reconstructed game states. Prompting-based; no RL or search.

**Memory Mechanism:**

> Game state is reconstructed externally from day/night cycles and public game logs and supplied as context; models do not maintain a learned memory store. The benchmark provides the history; the model reasons over it per query.

**Belief State Representation:**

> Natural-language reasoning over provided context. Beliefs about hidden roles are probed through an Opponent-role Inference task that compares the model's predicted opposing-faction set against ground-truth roles; there is no explicit probability-table belief state.

**Planning Approach:**

> No agentic planning loop. 'Planning' is evaluated only through Vote Alignment, which checks whether the model's elimination vote matches the winning faction's (MVP's) actual vote in the curated game.

**Action Tool Design:**

> No callable tools. The benchmark exposes structured tasks: 9-way multiple-choice speech-evaluation items and decision-evaluation items (vote choice, opponent-role set prediction); the model emits answers, not tool calls.

**Theory Of Mind Evaluation:**

> Yes, substantially. WereAlign explicitly assesses ToM-related abilities: Role Inference (uncovering true identities/intentions), Deception Reasoning (identifying lies and effective masquerade), and Opponent-role Inference, measuring how well models model other players' hidden roles and intent.


### Game Loop Design

**Turn Structure:**

> Reconstructed Day/Night cycles from the source games: day discussion with sequential speeches and a collective vote, night skill activations. The dataset spans 80+ games and 240+ day-night cycles.

**Phase Handling:**

> Phases (day discussion, day voting, night skill activation) are reconstructed from public game logs and host narration; the benchmark presents tasks anchored to specific phases rather than running a live game engine.

**Communication Protocol:**

> Public sequential speeches during limited speaking time plus collective voting, taken from real televised gameplay with real-time narration; private night actions are reconstructed into the game log.

**Communication Synchrony:**

> Synchronous, turn-based: players speak in sequence within limited speaking time and then vote collectively, as in the televised human format. No bidding or scheduler mechanism.


### Evaluation

**Metrics:**

> Speech Accuracy as a macro-average over five social-ability dimensions (Role Inference, Strategic Judgment, Deception Reasoning, Persuasive Statements, Counterfactual Trade-off); Decision metrics: Vote Alignment and Opponent-role Inference accuracy. Scores averaged over 5 independent runs.

**Deception Metric Design:**

> Deception is quantified through the dedicated Deception Reasoning dimension: multiple-choice tasks asking whether the model can identify lies or recognize effective masquerade, scored as accuracy against winning-faction ground truth. No detector-accuracy surrogate or explicit lie taxonomy.

**Persuasion Modeling:**

> Yes. A Persuasive Statements dimension evaluates whether models can generate context-appropriate persuasive expressions; the paper reports LLMs handle this dimension reasonably well, but persuasion is measured, not optimized.

**Rating System:**

> None. No Elo/TrueSkill; performance is reported as multiple-choice accuracy with standardized decoding averaged over 5 runs.

**Competition Format:**

> Not a head-to-head competition. Models are independently scored on the same static benchmark tasks; no self-play or cross-play tournament.

**Llms Evaluated:**

> 16 models. Proprietary: GPT-5, GPT-5-mini, GPT-5-nano, Gemini-2.5-Pro, Gemini-2.5-Flash, DeepSeek-V3.1, DeepSeek-V3.2-Exp, DeepSeek-R1, GLM-4.5. Open-source: Qwen3-32B, Qwen3-30B-A3B, QwQ-32B, Llama-4-Scout-17B, Gemma-3-27B-IT, GPT-OSS-20B.

**Human Baseline Comparison:**

> Human strategies serve as the ground truth: the winning faction's (and MVP's) decisions in professionally curated, expert-narrated televised games define correct answers. No explicit numeric human accuracy score is reported; humans are the alignment target rather than a side-by-side competitor.

**Data Source Provenance:**

> Curated real human gameplay: 100+ hours of professionally produced televised Werewolf matches from the Panda Kill platform (YouTube/Bilibili), processed via ASR transcription (4.9% WER after correction), speaker diarization and game-state reconstruction, with high inter-annotator agreement (kappa 0.97 speaker attribution, 0.93 game logs).


### Training Methodology

**Training Paradigm:**

> Evaluation only; tuning-free. No model training or fine-tuning is performed; the contribution is a dataset (WereBench) and an evaluation framework (WereAlign).

**Agent Controllability:**

> Not applicable; the work evaluates fixed models and does not tune agents to a target win rate or difficulty.

**Strategy Space Analysis:**

> Yes, descriptively. The paper analyzes human winning-faction strategies as ground truth and finds LLMs produce shallow, formulaic statements rather than authentic deception; it also notes Werewolves can hide by echoing the good faction while contributing minimally.


### Optimization

**Prompt Optimizer Used:**

> None. No DSPy or automated prompt optimizer; the paper notes simple prompt interventions (rule reminders, objective speech rewriting) help weaker models, but these are manual.

- **Rl Algorithm:** None. No RL or preference optimization.

**Reward Design:**

> No training reward. Scoring uses accuracy against winning-faction human strategy as ground truth; this is an evaluation signal, not a training reward.

- **Optimization Target:** Nothing is algorithmically optimized; the work is a benchmark and dataset, not a training method.

**Self Play Loop:**

> No. The benchmark explicitly criticizes LLM-vs-LLM self-play for producing templated utterances and instead uses curated human gameplay.

**Credit Assignment:**

> Not applicable in a training sense. Per-task scoring attributes correctness to individual speech or vote decisions, but there is no multi-turn credit assignment over delayed outcomes.


### Engineering

**Framework Used:**

> Custom evaluation framework (WereAlign) over the WereBench dataset; relies on an ASR/diarization/game-state-reconstruction annotation pipeline. No agent framework like DSPy or LangChain is used.

**Reproducibility:**

> Moderate-to-high in methodology: standardized decoding, 5-run averaging and high inter-annotator agreement (kappa 0.93-0.97) are documented; however, several evaluated models are closed APIs and no explicit reproducibility repository link is confirmed.


### Findings

**Key Results:**

> WereBench provides a high-quality, human-verified multimodal Werewolf dataset (100+ hours of video, 32.4M utterance tokens, 80+ games, 240+ day-night cycles, 15 rule variants, 30 roles). WereAlign uses winning-faction human strategy as ground truth across five speech dimensions and two decision tasks. State-of-the-art LLMs show diverse but generally weak performance: roughly half score below 0.50 average speech accuracy; the best model (Gemini-2.5-Pro) reaches about 0.720, while small models such as GPT-5-nano (0.317) are near random. Clear gaps appear in deception and counterfactual reasoning; rule reminders and objective speech rewriting help weaker models.

**Identified Limitations:**

> The dataset is drawn from one specific televised program and may not represent broader, casual gameplay styles; evaluation does not yet capture other social-intelligence aspects such as long-term cooperation; results are bounded by currently available systems and inference settings.

**Deception Emergence:**

> Findings on deception are largely negative: LLMs tend to generate shallow, formulaic statements rather than authentic, context-grounded deception, and roughly half of models score below 0.50 on deception-related reasoning. Werewolves can superficially hide by echoing the good faction while contributing little substantive content.

**Deceiver Detector Asymmetry:**

> The paper highlights that models struggle with the Deception Reasoning dimension (identifying lies/masquerade) while handling persuasive statements reasonably well, implying weakness on the detection side; it does not, however, run a controlled deceiver-vs-detector comparison.


### Uncertain Fields

- Open Source
- Optimization Cost
- Repo Url

---

## 9. Community LLM Werewolf/Mafia Arenas — MafiaBench and Foaster.ai Werewolf Arena

### Basic Info

- **Name:** Community LLM Werewolf/Mafia Arenas — MafiaBench and Foaster.ai Werewolf Arena
- **Year:** 2025
- **Paper Url:** https://www.mafiabench.org/ ; https://werewolf.foaster.ai/
- **Repo Url:** https://github.com/nickslevine/mafiabench ; https://github.com/Foaster-ai/Werewolf-bench

### Game Coverage

**Games Supported:**

> MafiaBench: Mafia (a.k.a. simple Werewolf), with no special town roles. Foaster.ai Werewolf Arena: Werewolf with special roles (Seer, Witch) plus a pre-game mayor election

**Num Players:**

> MafiaBench: 8 players (2 mafia, 6 townspeople). Foaster.ai: 6 players (2 werewolves, 4 villagers including 1 Seer and 1 Witch)

**Hidden Role Mechanic:**

> Hidden-role asymmetric information: mafia/werewolves know each other's identities and act covertly at night, while townspeople/villagers have no knowledge of who is mafia and must deduce identities from discussion and voting. Foaster.ai adds informed special roles (Seer can investigate, Witch has single-use potions)

**Rule Variant Coverage:**

> Two distinct community setups: MafiaBench uses a minimal roleless Mafia variant; Foaster.ai uses a richer Werewolf variant with Seer, Witch, and a mayor-election mechanic. Limited rule-variant breadth within each arena

- **Multimodal Support:** No; both arenas are text-only with no video, audio, or voice-prosody modalities

### Agent Architecture

**Theory Of Mind Evaluation:**

> Not measured as an explicit ToM accuracy metric; the arenas measure behavioral outcomes (manipulation success, vote-swing, suspicion resistance) that depend on modeling other players, and Foaster.ai uses role-conditioned Elo to separate manipulation from manipulation-resistance


### Game Loop Design

**Turn Structure:**

> Round-based alternation of day and night; within the day, players take sequential public speaking turns followed by a voting turn; at night, the mafia/werewolves (and special roles in Foaster.ai) act secretly

**Phase Handling:**

> Discrete day (discussion + voting) and night (secret elimination, and Seer/Witch actions in Foaster.ai) phases managed by the arena game engine; Foaster.ai adds a pre-game mayor-election phase

**Communication Protocol:**

> Public natural-language chat during day discussion plus public votes; private/secret channels for mafia coordination and night actions


### Evaluation

**Metrics:**

> Elo rating from head-to-head play; win rates split by side (mafia/town, wolf/villager). Foaster.ai additionally measures manipulation success via per-message vote-swing, wolf-side betrayal patterns, and Day-1 role-elimination rates

**Persuasion Modeling:**

> Persuasive influence is explicitly measured (Foaster.ai's per-message vote-swing / manipulation-success metric) but not optimized; models are ranked, not trained on persuasion

**Rating System:**

> MafiaBench: Elo computed via a 15-round Swiss tournament (pairing models by nearest Elo). Foaster.ai: Elo computed from a round-robin, with role-conditioned Elo (ELO-W for wolves, ELO-V for villagers)

**Competition Format:**

> Cross-play tournaments between different LLMs. MafiaBench: 15-round Swiss, 10 games per pairing per round (5 as mafia, 5 as town), 450 games total. Foaster.ai: round-robin with 10 matches per LLM pair across 7 models

**Llms Evaluated:**

> MafiaBench: ChatGPT-4o Latest, Gemini Pro 1.5, Gemini 2.0 Flash, Claude 3.5 Sonnet, Claude 3.7 Sonnet, GPT-4o Mini. Foaster.ai: GPT-5, GPT-5-mini, GPT-OSS-120B, Gemini 2.5 Pro, Gemini 2.5 Flash, Grok-4 / Grok-4-fast-reasoning, Kimi-K2 / Kimi-K2-0905-preview, Qwen3-235B

**Human Baseline Comparison:**

> No human players; both arenas pit LLMs only against other LLMs and provide no direct human baseline comparison

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay generated and logged by the arena engines; continuously updated as new models are added


### Training Methodology

- **Training Paradigm:** Tuning-free; arenas evaluate off-the-shelf models without any SFT, RL, or DPO
- **Agent Controllability:** Not applicable; the arenas do not tune agent skill to a target win rate or difficulty

**Strategy Space Analysis:**

> Informal/qualitative analysis of emergent behaviors (e.g., wolf-side betrayal patterns, Day-1 eliminations, stated-vs-actual strategy divergence) rather than formal strategy-space expansion


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method is used
- **Rl Algorithm:** None; no RL or preference-optimization algorithm is used

**Reward Design:**

> Not applicable; no training signal. Game outcome (win/loss) feeds the Elo update, not a learning reward

- **Optimization Target:** Not applicable; nothing is optimized — these are leaderboards/benchmarks, not training systems

**Self Play Loop:**

> Repeated LLM-vs-LLM games are played to populate the leaderboard, but there is no iterative self-play training; gameplay is used only for rating

**Credit Assignment:**

> Not applicable; no learning credit assignment. Game outcomes are attributed at the side/model level for Elo and win-rate computation


### Engineering

**Open Source:**

> Yes; both arena codebases are open source on GitHub, and leaderboards are publicly hosted and continuously updated


### Findings

**Key Results:**

> MafiaBench (450-game Swiss tournament, 8-player Mafia): ChatGPT-4o Latest leads with ~1740 Elo, followed by Gemini Pro 1.5 (~1627) and Gemini 2.0 Flash (~1561). Foaster.ai Werewolf Arena (round-robin of 7 LLMs, 6-player Werewolf): GPT-5 leads the Elo leaderboard clearly while GPT-OSS sits at the bottom; the remaining models form a close second pack whose strength depends on role, motivating role-conditioned Elo to separate wolf manipulation from villager manipulation-resistance

**Deception Emergence:**

> Both arenas observe emergent deception and persuasion: mafia/wolf models successfully manipulate voting (measurable vote-swing in Foaster.ai), exhibit betrayal patterns among wolves, and the gap between stated and actual private-thought strategy reveals deliberate concealment. Win-rate-as-mafia serves as direct evidence of effective deception


### Uncertain Fields

- Action Tool Design
- Authors Org
- Belief State Representation
- Communication Synchrony
- Deceiver Detector Asymmetry
- Deception Metric Design
- Framework Used
- Identified Limitations
- Memory Mechanism
- Optimization Cost
- Planning Approach
- Reasoning Paradigm
- Reproducibility

---

## 10. DVM: Towards Controllable LLM Agents in Social Deduction Games

### Basic Info

- **Name:** DVM: Towards Controllable LLM Agents in Social Deduction Games
- **Year:** 2025

**Authors Org:**

> Zheng Zhang, Yihuai Lan, Yangsen Chen, Lei Wang, Xiang Wang, Hao Wang — The Hong Kong University of Science and Technology (Guangzhou); Singapore Management University; University of Science and Technology of China. Published at ICASSP 2025

- **Paper Url:** https://arxiv.org/abs/2501.06695

### Game Coverage

- **Games Supported:** Werewolf (one of the most popular social deduction games)
- **Num Players:** 9 players: 3 werewolves, 1 seer, 1 witch, 1 hunter, and 3 villagers

**Hidden Role Mechanic:**

> Hidden roles split into the werewolf camp and the village camp; werewolves know each other and act covertly at night, while informed special roles (seer can check identities, witch can save/poison, hunter) and plain villagers must deduce werewolf identities through day discussion and voting

**Rule Variant Coverage:**

> Single 9-player Werewolf ruleset with a fixed role composition; no multiple role-set variants are explored

- **Multimodal Support:** No; text-based Werewolf only, with no video, audio, or voice-prosody modalities

### Agent Architecture

**Reasoning Paradigm:**

> Modular RL-augmented architecture: an LLM-based Predictor for identity inference, a neural Decider policy (encoder-decoder with masked softmax over legal actions) trained with PPO, and an LLM-based Discussor for dialogue generation; baselines include ReAct, Least-to-Most, and Thinker

**Belief State Representation:**

> Beliefs are produced by the Predictor as formatted natural-language predictions of other players' identities (e.g., 'Player 1: Seer, Player 2: Werewolf'), which are then parsed and vectorized to feed the Decider; effectively a discrete per-player role prediction rather than a continuous probability table

**Planning Approach:**

> No explicit search; planning is implicit in the Decider's learned policy and in the decision chain reward that evaluates entire decision sequences (chains) across a game rather than single steps

**Action Tool Design:**

> Game actions (discuss, vote, kill, save, check, poison) are exposed as a discrete action space; the Decider encodes events via subject/verb/object embedding layers and outputs action logits with an illegal-action mask applied through masked softmax to enforce game rules per phase

**Theory Of Mind Evaluation:**

> Identity inference is explicitly measured as a proxy for ToM: Werewolf Prediction (identify 3 werewolves among 8 others) and Identity Prediction (predict all 8 others' roles), reported as ACC@N; no second-order belief modeling


### Game Loop Design

**Turn Structure:**

> Phase-based: a day phase with sequential discussion turns followed by voting, and a night phase with secret role actions (kill, save, check, poison)

**Phase Handling:**

> Distinct day (discuss + vote) and night (kill/save/check) phases handled by the Werewolf game environment; the Decider's legal-action mask is phase-dependent

**Communication Protocol:**

> Public natural-language discussion and public votes during the day; secret night actions; the Discussor generates persuasive/deceptive/informative speech based on role and strategy


### Evaluation

**Metrics:**

> Win rate by camp (werewolf / villager / other roles); identity-prediction accuracy ACC@N for Werewolf Prediction and Identity Prediction; and, for controllability, the gap between achieved win rate and the specified win-rate constraint

**Deception Metric Design:**

> No dedicated deception metric or lie taxonomy; deceptive ability is reflected indirectly through werewolf-camp win rate and through the Discussor's role-conditioned persuasive/deceptive speech generation

**Persuasion Modeling:**

> The Discussor explicitly generates speech intended to persuade, deceive, or inform other agents based on the agent's strategy and role, but persuasive influence is not separately quantified or optimized as its own objective

**Rating System:**

> No Elo/TrueSkill; head-to-head win rates measured against the Thinker baseline across 30 games per setting

**Competition Format:**

> Cross-play: a target method controls one camp while the remaining roles are controlled by the Thinker baseline; 30 games per configuration

**Llms Evaluated:**

> ChatGLM3-6B is the base model for the Predictor and Discussor; baselines compared include GPT-3.5, GPT-4, ReAct, Least-to-Most (LtM), and Thinker

**Human Baseline Comparison:**

> No direct head-to-head play against humans; however, supervised training and the decision-chain win-rate database are derived from the FanLang-9 dataset of over 18,000 human-player game records

**Data Source Provenance:**

> Curated real human gameplay (FanLang-9, 18,000+ human Werewolf game records) for supervised training and decision-chain statistics, plus synthetic self-play games generated during RL and for evaluation (600 games sampled for the prediction test set)


### Training Methodology

**Training Paradigm:**

> Two-stage: supervised fine-tuning on the FanLang-9 human dataset, followed by reinforcement learning via self-play — PPO for the Decider and DPO for the Predictor

**Agent Controllability:**

> Yes — the central contribution. A win-rate-constrained reward lets the agent be tuned toward a specified target win rate; experiments sweep constraints of 10/30/50/70/90% and show DVM's actual win rate trends upward with the target (clear controllability) while baselines are insensitive to prompted constraints

**Strategy Space Analysis:**

> No formal strategy-space-expansion analysis; the decision-chain reward analyzes which sequences of decisions correlate with higher win rates, and ablations isolate the Predictor and decision-chain reward


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method is used

**Rl Algorithm:**

> PPO for the Decider policy and DPO for the Predictor (predictions vs. correct answers used as negative/positive preference pairs)

**Reward Design:**

> Per-step reward r_t = sr_t (step reward) + cr (chain reward). The decision-chain reward cr(DC) = alpha*(WR-0.5) is derived from a database of (decision-chain, win-rate) pairs mined from FanLang-9. For controllable agents the chain reward is replaced by a win-rate-constrained reward cr_ctrl that uses the squared deviation between target and achieved win rate, a threshold epsilon, and a tanh-based scaling into [-s, s] to push the agent toward the target win rate

**Optimization Target:**

> Policy weights — the Decider (PPO) and the Predictor (DPO) networks; the Discussor LLM is not trained. Optimization improves decision chains and identity prediction, not prompts

**Self Play Loop:**

> Yes; after supervised training, the Predictor and Decider are further optimized through self-play during the RL phase

**Credit Assignment:**

> Addressed via the decision-chain reward: instead of crediting only single-step actions, the entire game-level decision chain is scored against a win-rate database, distributing delayed game-outcome credit across the sequence of decisions; PPO's advantage A = r_t + gamma*V(s_t+1) - V(s_t) handles temporal credit


### Findings

**Key Results:**

> DVM both outperforms prior methods and achieves controllable performance. On identity prediction it leads all baselines (Werewolf Prediction ACC@1 0.908 vs Thinker 0.853; ACC@3 0.090; Identity Prediction ACC@1 0.972, ACC@5 0.170). In win-rate comparison against Thinker it reaches 66.6% as werewolf, 63.3% as villager, and 53.3% as other roles, beating ReAct, GPT-3.5, GPT-4, LtM, and Thinker. Ablations show drops without the decision-chain reward (villager 63.3->56.6%) and without the Predictor (63.3->46.0%). For controllability, DVM's actual win rate rises monotonically with the specified constraint (10-90%) while ReAct, LtM, and Thinker stay flat regardless of prompted constraints, though a gap remains between achieved and target win rates and high targets are hard to reach

**Deception Emergence:**

> Deceptive behavior emerges implicitly through the Discussor generating role-conditioned persuasive/deceptive speech and through the werewolf camp's strong win rate (66.6%), but the paper does not explicitly catalog or measure forms of emergent deception


### Uncertain Fields

- Communication Synchrony
- Deceiver Detector Asymmetry
- Framework Used
- Identified Limitations
- Memory Mechanism
- Open Source
- Optimization Cost
- Repo Url
- Reproducibility

---

## 11. LSPO: Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization

### Basic Info

**Name:**

> LSPO: Learning Strategic Language Agents in the Werewolf Game with Iterative Latent Space Policy Optimization

- **Year:** 2025

**Authors Org:**

> Zelai Xu, Wanjun Gu, Chao Yu, Yi Wu, Yu Wang — Tsinghua University; Beijing Zhongguancun Academy; Shanghai Qi Zhi Institute. Published at ICML 2025 (PMLR vol. 267)

- **Paper Url:** https://arxiv.org/abs/2502.04686

### Game Coverage

**Games Supported:**

> Werewolf (a text-based social deduction game), with a Rock-Paper-Scissors-Spock-Lizard toy game used only as a proof-of-concept illustration

**Num Players:**

> 7-player Werewolf (2 Werewolves, 1 Seer, 1 Doctor, 3 Villagers); a simpler 4-player variant (1 Werewolf, 1 Seer, 2 Villagers) is used for ablation sensitivity analysis

**Hidden Role Mechanic:**

> Hidden roles split into a Werewolf side and a Village side; the two Werewolves know each other and try to eliminate others while hiding, the Seer can investigate one player's role each night, the Doctor can protect one player each night, and Villagers have no ability — all communicating purely through free-form natural language

**Rule Variant Coverage:**

> Primarily one 7-player Werewolf ruleset; one additional simplified 4-player variant for ablations. Limited rule-variant breadth — the focus is depth of strategy learning rather than cross-variant generalization

**Multimodal Support:**

> No; a deliberately pure text-based environment that explicitly excludes speaking tone, facial expression, and body language


### Agent Architecture

**Reasoning Paradigm:**

> Game-theoretic policy learning over an abstracted latent strategy space: free-form utterances are clustered into discrete latent strategies, Counterfactual Regret Minimization (CFR, scaled via neural networks / Deep CFR) solves the abstracted extensive-form game, and the LLM is fine-tuned to the learned latent policy; baselines compared include ReAct, ReCon, a Cicero-like agent, and SLA

**Planning Approach:**

> Latent-space planning: the free-form game is reformulated as an abstracted extensive-form game over discrete latent strategies, and CFR is run in that latent space to compute near-optimal policies; there is no per-turn explicit tree search at inference

**Action Tool Design:**

> Player actions are natural language of three types — secret actions (night eliminate/investigate/protect), discussion actions (day statements), and voting actions; free-form discussion utterances are abstracted into a finite set of latent strategies (via LLM generation + k-means clustering of text embeddings) that serve as the tractable action set for game-theoretic optimization, while secret and voting actions are already discrete

**Theory Of Mind Evaluation:**

> ToM is evaluated implicitly via a role-prediction phase added before voting; prediction accuracy for Werewolf, Seer, Doctor, and Villager roles is reported per LSPO iteration, but there is no explicit second-order belief-modeling metric


### Game Loop Design

**Turn Structure:**

> Alternating night and day rounds: in the night round players perform secret actions; in the day round players make sequential discussion statements and then cast votes

**Phase Handling:**

> Distinct night phase (secret eliminate/investigate/protect actions) and day phase (discussion then voting) handled by the text-based Werewolf environment

**Communication Protocol:**

> Public free-form natural-language discussion plus public voting during the day; secret private night actions for Werewolves, Seer, and Doctor


### Evaluation

**Metrics:**

> Win rate (by side and overall) and role-prediction accuracy per iteration; for the toy game, action-distribution exploitability relative to the Nash equilibrium

**Deception Metric Design:**

> No dedicated deception metric or lie taxonomy; deception is observed qualitatively through the evolving latent strategy space (e.g., bluffing, pretending to be Seer, fabricating investigation results) and reflected in Werewolf-side win rate

**Persuasion Modeling:**

> Persuasive/strategic discussion is learned and analyzed (e.g., the Seer learns voting-coordination strategies; the Werewolf learns misdirection) but persuasive influence is not separately quantified as its own optimization objective

**Rating System:**

> No Elo/TrueSkill; head-to-head win rates against a fixed final-iteration LSPO opponent and against four state-of-the-art baselines, averaged over 100 games per setting

**Competition Format:**

> Self-play during training plus cross-play evaluation: the final-iteration LSPO agent is used as a fixed opponent while LSPO agents from different iterations, and four baseline agents, play 100 Werewolf games as each side

**Llms Evaluated:**

> Llama-3-8B-Instruct as the base/backbone model fine-tuned by LSPO; gpt-3.5 is used to generate preference data for the 'w/o policy learning' ablation; baselines (ReAct, ReCon, Cicero-like, SLA) are also LLM-based agents

- **Human Baseline Comparison:** No direct head-to-head play against humans; evaluation is entirely LLM-agent-vs-LLM-agent

**Data Source Provenance:**

> Synthetic LLM self-play gameplay generated by the LSPO agent across iterations; no human gameplay data is used


### Training Methodology

**Training Paradigm:**

> Iterative framework combining CFR-in-latent-space game-theoretic optimization with LLM fine-tuning via DPO; the base LLM is fine-tuned (not tuning-free)

**Agent Controllability:**

> Not a goal; LSPO targets maximal strategic performance rather than tuning the agent to a specified win rate or difficulty

**Strategy Space Analysis:**

> Yes — a central theme. The latent strategy space is visualized across iterations (t-SNE-style projections for Werewolf and Seer roles), showing progressively clearer, more refined clusters and increasingly sophisticated strategies (bluffing, misdirection, voting coordination); the framework explicitly performs latent-space expansion across iterations


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method is used — improvement comes from CFR + DPO fine-tuning

**Rl Algorithm:**

> Counterfactual Regret Minimization (CFR), implemented as Deep CFR with neural networks to approximate regret, used as the game solver in latent space; the LLM is then aligned to the learned policy with Direct Preference Optimization (DPO)

**Reward Design:**

> Game-theoretic: CFR minimizes counterfactual regret per information set (R_t(a) = R_t-1(a) + u(sigma_t^a, sigma_t^-a) - u(sigma_t)) from game-outcome utilities. The DPO preference dataset is constructed from regret values — a discussion candidate mapped to a lower-regret latent strategy is preferred over a higher-regret one

**Optimization Target:**

> Both a policy and the LLM weights: CFR learns a latent-space policy from the abstracted game, then DPO fine-tunes the base LLM (Llama-3-8B-Instruct) to align its free-form outputs with that policy; the latent strategy space is also expanded each iteration

**Self Play Loop:**

> Yes; iterative self-play drives the loop — each iteration generates latent strategies via LLM self-play across roles, solves the abstracted game, fine-tunes the LLM, then re-generates and expands the latent space using the fine-tuned model

**Credit Assignment:**

> Multi-turn dialogue credit is assigned game-theoretically: discussion utterances are abstracted to latent strategies forming information sets in an extensive-form game, and CFR's counterfactual regret distributes delayed game-outcome utility across those decision points; DPO preferences are then derived from the resulting regret values


### Findings

**Key Results:**

> LSPO iteratively expands the strategy space with improving performance and outperforms state-of-the-art Werewolf agents. Against a fixed final-iteration LSPO opponent, win rate and role-prediction accuracy rise monotonically across iterations: Werewolf-side win rate 0.54 -> 0.63 -> 0.73 and Village-side 0.18 -> 0.23 -> 0.27 over iterations 1-3, with overall prediction accuracy rising to 0.83 (Werewolf side) and 0.73 (Village side). Against four baselines the final LSPO agent achieves the highest win rates: 0.73 as Werewolf and 0.27 as Village (0.50 overall), beating ReAct (0.38), ReCon (0.38), Cicero-like (0.44), and SLA (0.47). In the Rock-Paper-Scissors-Spock-Lizard proof-of-concept, LSPO reaches the Nash equilibrium (exploitability 0.00) after 3 iterations while ReAct and SLA do not. Ablations confirm both policy learning and LLM fine-tuning are necessary, and results are robust to cluster size k and DPO beta

**Deception Emergence:**

> Clear emergent deception: the visualized latent strategy space shows the Werewolf agent abandoning a naive role-revealing strategy and developing deliberate bluffs and misdirection (e.g., pretending to be the Seer and providing fabricated investigation results to sow confusion, defending teammates and redirecting suspicion onto scapegoats), with the strategy set growing more sophisticated each iteration


### Uncertain Fields

- Belief State Representation
- Communication Synchrony
- Deceiver Detector Asymmetry
- Framework Used
- Identified Limitations
- Memory Mechanism
- Open Source
- Optimization Cost
- Repo Url
- Reproducibility

---

## 12. MaKTO — Multi-agent Kahneman-Tversky Optimization (Werewolf/Mafia)

### Basic Info

- **Name:** MaKTO — Multi-agent Kahneman-Tversky Optimization (Werewolf/Mafia)
- **Year:** 2025

**Authors Org:**

> Rong Ye, Yongxin Zhang, Yikai Zhang, Haoyu Kuang, Zhongyu Wei, Peng Sun (Fudan University DISC Lab and collaborators). Published at NeurIPS 2025.

- **Paper Url:** https://arxiv.org/abs/2501.14225
- **Repo Url:** https://reneeye.github.io/MaKTO.html

### Game Coverage

**Games Supported:**

> Werewolf (also referred to as Mafia), a social deduction game; specifically the 9-player Seer-Witch-Guard variant. The method is presented as generalizable to language games more broadly.

**Num Players:**

> 9-player Werewolf games (3 werewolves, 6 villagers including special roles Seer, Witch, Guard/Hunter).

**Hidden Role Mechanic:**

> Standard Werewolf hidden-role structure: werewolves secretly know each other and act at night; villagers and special roles (Seer, Witch, Guard) have asymmetric private information. Players infer hidden identities from speech and voting patterns.

**Rule Variant Coverage:**

> Primarily one rule variant (9-player Seer-Witch-Guard). The paper notes a Hunter role generalizes, but coverage of distinct rule sets is narrow — essentially a single Werewolf configuration.

- **Multimodal Support:** Text-only; no video, audio, or voice-prosody modalities are handled.

### Agent Architecture

**Reasoning Paradigm:**

> Unified language-action LLM policy: decision-making and natural-language generation are integrated in a single model (no separate policy vs. dialogue modules). Reasoning is in-context, supported by annotated thinking-process data, refined via preference optimization (KTO).

**Memory Mechanism:**

> Implicit memory through extended conversational context within the LLM; the model maintains game-state understanding via in-context learning over the dialogue history rather than an explicit external memory store or summary module.

**Belief State Representation:**

> No structured belief state. Agents infer other players' likely roles from speech analysis and voting patterns expressed in natural language; beliefs are latent in the LLM rather than a probability table.

**Planning Approach:**

> No explicit planning module. Strategy is encoded implicitly via annotated thinking-process data (actions, speeches, voting rationales) and learned through behavior cloning plus KTO refinement.

**Action Tool Design:**

> Game actions (night kill, Seer check, Witch save/poison, Guard protect, day speech, voting) are produced as structured natural-language outputs by the unified LLM; not exposed as separate callable tools/functions.


### Game Loop Design

**Turn Structure:**

> Turn-based: within each round players take ordered turns for night actions, day speeches, and voting.

**Phase Handling:**

> Standard Werewolf day/night cycle: night phase (werewolf kill, Seer check, Witch potions, Guard protect), day discussion phase (sequential speeches), and voting phase to eliminate a player.

**Communication Protocol:**

> Public day-phase discussion via sequential speeches plus voting; private night actions for werewolves and special roles. Communication is structured turn-based natural language.

**Communication Synchrony:**

> Synchronous turn-based: players speak in a fixed order rather than via asynchronous bidding or scheduler-driven free-form speech. The paper lists turn-based (non-free-form) communication as a limitation.


### Evaluation

**Metrics:**

> Win rate (primary, 61% average in 9-player games), role-specific identification accuracy (Seer werewolf-ID 75.7%), win rate against human experts (60%), and Turing-style human-recognition rate (48.9% detectability).

**Deception Metric Design:**

> Deception is quantified via opponent-confusion metrics: higher abstention rates and lower role-identification accuracy induced in opponents, plus the Turing-style blind test where 48.9% detectability indicates convincing identity masking.

**Persuasion Modeling:**

> Persuasive influence is captured indirectly through staged voting-based preference selection — speeches are evaluated by their effect on subsequent voting outcomes — but persuasion is not modeled as an explicit standalone objective.

**Competition Format:**

> Cross-play: agents play in a diverse model pool (GPT-4o, GPT-4o-mini, Claude-3.5, fine-tuned Qwen/Llama variants) to generate data and for evaluation; also mixed human-AI games. Multi-agent gameplay is preferred over pure self-play to avoid strategy fixation.

**Llms Evaluated:**

> Tested/compared: GPT-4o, GPT-4o-mini, Claude-3.5-Sonnet, fine-tuned Llama-3.1-8B-Instruct. Training base models: Qwen2.5-14B-Instruct and Qwen2.5-72B-Instruct.

**Human Baseline Comparison:**

> Yes — MaKTO was evaluated against 14 expert human players (each with 1000+ games of experience) in organized mixed human-AI games, achieving a 60% win rate; a Turing-style blind test gave 48.9% detectability versus 76.6% for GPT-4o.

**Data Source Provenance:**

> Mixed: behavior-cloning data from game jargon, strategy guides and expert human annotations; preference data generated synthetically from multi-agent LLM-vs-LLM gameplay; evaluation also includes LLM-vs-human games.


### Training Methodology

**Training Paradigm:**

> Three-stage pipeline: (1) supervised behavior cloning (25k SFT samples), (2) multi-agent gameplay data collection (20k preference samples), (3) Kahneman-Tversky Optimization (KTO) refinement (12k desirable + 8k undesirable samples).

**Strategy Space Analysis:**

> Yes — the paper analyzes strategy fixation, showing multi-agent gameplay (61% win rate) outperforms self-play KTO (57%) by promoting more diverse, generalizable strategies.


### Optimization

**Prompt Optimizer Used:**

> None — MaKTO uses model weight optimization (KTO), not DSPy/prompt optimizers such as GEPA or MIPROv2.

**Rl Algorithm:**

> Kahneman-Tversky Optimization (KTO), a prospect-theory-based preference optimization algorithm that uses unpaired desirable/undesirable responses (loss weights lambda_D=0.7, lambda_U=1.0) instead of paired preferences as in DPO.

**Reward Design:**

> Step-wise (not whole-trajectory) preference labels rather than a scalar reward. Three complementary fine-grained selection methods: heuristic-based selection (game-specific good/bad actions), staged voting-based selection (speeches scored by voting outcomes), and verifier-based selection (fact-consistency checks via strong LLMs such as Claude and GPT-4o).

**Optimization Target:**

> Policy weights of the LLM agent (decision-making and language generation jointly), refined via KTO over step-wise preference-labeled responses.

**Self Play Loop:**

> Uses multi-agent gameplay with a diverse model pool rather than pure self-play; the paper explicitly shows multi-agent data (61%) beats self-play KTO (57%) by avoiding strategy fixation.

**Credit Assignment:**

> Addresses sparse delayed credit by abandoning whole-trajectory win/loss labeling in favor of step-wise preference selection, assigning desirable/undesirable labels to individual actions/speeches via heuristics, voting-outcome signals, and LLM verifiers.


### Engineering

**Framework Used:**

> Custom three-stage training pipeline (SFT behavior cloning, multi-agent gameplay data generation, KTO refinement) built on Qwen2.5 base models; not built on DSPy or LangChain.

**Open Source:**

> Code and data announced as to be released via the project page (https://reneeye.github.io/MaKTO.html).


### Findings

**Key Results:**

> In 9-player Werewolf, MaKTO achieves 61% average win rate, beating GPT-4o (50%, +23.0% relative) and a two-stage Mix-RL agent (55%, +10.9% relative). Against 14 expert human players it wins 60% of games, and in a Turing-style blind test it shows only 48.9% detectability (vs 76.6% for GPT-4o). Seer role werewolf-identification reaches 75.7% vs 60% SFT baseline. Multi-agent KTO (61%) outperforms self-play KTO (57%).

**Identified Limitations:**

> Turn-based rather than free-form communication; occasional hallucinations and inconsistency in extended conversations; offline learning paradigm (online RL unexplored); confined to specific Werewolf variants.

**Deception Emergence:**

> MaKTO agents exhibit sophisticated learned deception: werewolf agents convincingly mask their identity through strategic speech and voting manipulation, raising opponent abstention rates and lowering opponents' role-identification accuracy; the 48.9% Turing detectability indicates human-like deceptive play.


### Uncertain Fields

- Agent Controllability
- Deceiver Detector Asymmetry
- Optimization Cost
- Rating System
- Reproducibility
- Theory Of Mind Evaluation

---

## 13. MultiMind: Enhancing Werewolf Agents with Multimodal Reasoning and Theory of Mind

### Basic Info

- **Name:** MultiMind: Enhancing Werewolf Agents with Multimodal Reasoning and Theory of Mind
- **Year:** 2025

**Authors Org:**

> Zheng Zhang, Nuoqian Xiao, Qi Chai, Deheng Ye, Hao Wang — The Hong Kong University of Science and Technology (Guangzhou) and Tencent. Published at ACM Multimedia (MM '25)

- **Paper Url:** https://arxiv.org/abs/2504.18039
- **Repo Url:** https://github.com/CjangCjengh/onuw

### Game Coverage

**Games Supported:**

> One Night Ultimate Werewolf (ONUW), a variant of Werewolf with role changes and a single night phase followed by one day phase

**Num Players:**

> 5 players (1 werewolf, 1 seer, 1 robber, 1 troublemaker, 1 insomniac); werewolf forms Team Werewolf, the rest form Team Village

**Hidden Role Mechanic:**

> Hidden roles with role-changing actions during the night: the werewolf is the hidden adversary, while special roles (seer, robber, troublemaker) act covertly at night and can alter who holds which role, increasing uncertainty; Team Village wins if the werewolf receives the most votes

**Rule Variant Coverage:**

> Single game (ONUW) at the hard difficulty setting of the Jin et al. environment; no multiple role-set variants are explored — generalization breadth is limited but the work supports both easy and hard difficulty levels of the same game

**Multimodal Support:**

> Yes — a core contribution. The Perceiver processes facial expressions from video (Emotion-LLaMA) and vocal tone from audio (OSUM), classifying each into 8 emotion categories alongside transcribed speech; the first framework to integrate multimodal information into social-deduction-game agents


### Agent Architecture

**Reasoning Paradigm:**

> Hybrid: a learned Theory-of-Mind Transformer model for belief inference, combined with Monte Carlo Tree Search (MCTS) for communication planning, and LLM-based perception and natural-language generation. A ReAct baseline is also implemented for comparison

**Memory Mechanism:**

> Game history is stored as a chronological sequence of structured action triplets (subject, predicate, object) plus per-time-step facial and vocal emotion labels; the full history (S_1:t, e_face_1:t, e_tone_1:t) is fed to the ToM Transformer (causal attention) and to the Actor LLM as context

**Belief State Representation:**

> Explicit second-order belief matrix B, where B[i,j] is the probability that player p_i believes player p_j is a werewolf; produced by a learned Transformer ToM model via a softmax linear head over causal hidden states, updated as new triplets/emotions arrive

**Planning Approach:**

> Explicit search: Monte Carlo Tree Search over a hierarchical tree whose first level chooses an (facial, vocal) emotion pair (64 nodes) and whose deeper levels incrementally build the action triplet sequence A_t+1 (max depth 3, 36 branches per node); UCT selection (C=1.414), 500 iterations, reward = negative summed suspicion toward the agent

**Action Tool Design:**

> Communication is decomposed into a finite predicate action space — 'support', 'suspect', and 'accuse as role' for each of 5 roles (2+5=7 predicates) — with subject/object over the player set; the Planner composes triplet sequences which the Actor LLM then realizes as natural-language utterances. Game decisions (votes, night actions) are also produced by the Actor

**Theory Of Mind Evaluation:**

> Explicitly central: the framework models second-order ToM (what others believe about oneself and others) via the belief matrix, and ToM model quality is measured by cross-entropy validation loss against ground-truth suspicion matrices; ablations isolate the ToM model's contribution


### Game Loop Design

**Turn Structure:**

> Within the day phase, all players engage in three rounds of sequential discussion turns, followed by a voting round; the night phase precedes the day with secret role actions

**Phase Handling:**

> Distinct night phase (secret role actions, role changes) and day phase (three discussion rounds + voting) managed by the 5-player ONUW environment of Jin et al. (KylJin/Werewolf)

**Communication Protocol:**

> Public natural-language statements during day discussion plus public votes; secret night actions are private; when playing humans, video and audio are also captured per player

**Communication Synchrony:**

> Synchronous, turn-based: discussion proceeds in fixed sequential rounds with each player speaking in turn before the voting step


### Evaluation

**Metrics:**

> Win rate (by side and overall) and average votes received per game (lower indicates better suspicion avoidance); ToM-model validation loss; planning time; for human studies, average human votes received

**Deception Metric Design:**

> Deception/suspicion-evasion is quantified by 'average votes received' (and 'average human votes' against humans) — fewer votes against the agent indicates more successful concealment; the MCTS objective directly minimizes summed suspicion B[j,i] toward the agent

**Persuasion Modeling:**

> Persuasive influence is explicitly optimized: the Planner searches for utterance sequences (and emotion labels) that minimize other players' suspicion toward the agent, i.e., it actively shapes others' beliefs

**Rating System:**

> No Elo/TrueSkill; head-to-head win-rate matrices across agent types (30 games per cell) and mixed-agent games (400 games with random agent selection)

**Competition Format:**

> Cross-play and mixed-play: agent-vs-agent matches where each team uses one agent type, plus 400 mixed games with 5 players randomly drawn from MultiMind and 4 baselines; plus a human user study

**Llms Evaluated:**

> GPT-4o, Qwen2.5-14B-Instruct, and Llama-3.1-8B-Instruct used to generate self-play training data; Gemini-2.0-Flash used as the default backend LLM in main experiments to demonstrate generalization

**Human Baseline Comparison:**

> Yes; a user study with 8 volunteers, each playing 10 games of 5-player ONUW alongside 4 randomly selected AI agents; multimodal perception (video/audio) is used to process human players, and human win rate (40.0%) is reported for comparison

**Data Source Provenance:**

> Both synthetic LLM self-play (5,300 generated 5-player ONUW games) and curated real human gameplay (the Lai et al. 'Werewolf Among Us' dataset: 163 videos / 199 games with timestamped transcripts and intention annotations)


### Training Methodology

**Training Paradigm:**

> Supervised training of a lightweight ToM model only (the LLMs are not fine-tuned): two-stage — cross-entropy training on self-play data, then fine-tuning on human gameplay data

**Agent Controllability:**

> Not addressed; the agent aims to maximize performance/suspicion-avoidance and is not tuned to a target win rate

**Strategy Space Analysis:**

> Qualitative discussion of emergent suspicion-minimizing communication strategies; ablations of MCTS iteration count, planner type (Random/DFS/BFS/MCTS), and multimodal inputs, but no formal strategy-space-expansion analysis


### Optimization

- **Prompt Optimizer Used:** None; no DSPy or prompt-optimization method is used
- **Rl Algorithm:** None; the ToM model is trained with supervised cross-entropy loss, not RL or preference optimization

**Reward Design:**

> Within MCTS, the search reward R is the negative sum of suspicion directed at the agent (R = -sum_j B_t+1[j,i]); the ToM model itself is trained with cross-entropy loss against ground-truth belief matrices, not a game-outcome reward

**Optimization Target:**

> A 25.4M-parameter Transformer ToM model (belief-prediction network); the LLM weights are not optimized — only belief modeling is learned

**Self Play Loop:**

> Yes for data generation: LLM agent self-play produces 5,300 games (5,000 train / 300 val) used to train the ToM model; it is single-stage data generation rather than iterative population-based training

**Credit Assignment:**

> ToM-model credit assignment is per-time-step supervised (belief matrix predicted and scored at each game event); MCTS handles multi-turn planning credit via backpropagation of the suspicion-based reward through the search tree, sidestepping delayed game-outcome credit

**Optimization Cost:**

> 5,300 self-play games generated in ~77 hours on 4 A800 GPUs; ToM model (25.4M params, hidden 512, 8 decoder layers) trained in ~4.5 hours on one A800 (80 epochs); human-data fine-tuning ~11 minutes (35 epochs)


### Engineering

**Framework Used:**

> Custom MultiMind framework (Perceiver, Reasoner, Planner, Actor) built on the 5-player ONUW environment of Jin et al. (github.com/KylJin/Werewolf); uses Emotion-LLaMA and OSUM for multimodal perception; no DSPy/LangChain

**Open Source:**

> Yes; code released at github.com/CjangCjengh/onuw, and it builds on the public ONUW environment and the public Lai et al. human dataset

**Reproducibility:**

> Good; the paper details model architecture, training hyperparameters, hardware, game counts, and the environment, and code plus the underlying datasets are public


### Findings

**Key Results:**

> MultiMind achieves the best performance among all baselines. In 400 mixed-agent games it reaches 70.9% win rate as Team Werewolf and 44.4% as Team Village (49.8% overall), with the lowest average votes received, beating ReAct, Belief, LLM-instructed, and RL-instructed agents. In head-to-head play it consistently tops the win-rate matrix. Ablations show MCTS beats Random/DFS/BFS planners (66.0% overall vs 57.0/44.0/61.0%), the lightweight ToM model matches an LLM-based reasoner at far lower cost, MCTS gains saturate around 500 iterations (51%->66% from 200->500, negligible 500->1000), and multimodal cues (facial + vocal emotion) lower ToM validation loss and improve win rates. In the human study MultiMind attains the highest win rate (42.2%) and the lowest human votes received (0.19), exceeding human players (40.0%)

**Deception Emergence:**

> Strong emergent deception: the agent learns communication strategies that actively minimize suspicion directed at itself, performing especially well as the werewolf (70.9% win rate) and receiving the fewest votes from both agents and humans, evidencing effective concealment and manipulation of others' beliefs through both text and chosen emotional expression


### Uncertain Fields

- Deceiver Detector Asymmetry
- Identified Limitations

---

## 14. The MindGames Challenge: Theory-of-Mind and Game Intelligence in LLM Agents (NeurIPS 2025 Competition)

### Basic Info

**Name:**

> The MindGames Challenge: Theory-of-Mind and Game Intelligence in LLM Agents (NeurIPS 2025 Competition)

- **Year:** 2025

**Authors Org:**

> Organizers: Kevin Wang, Jianzhu Yao, Yihan Jiang, Benjamin Finch, Viraj Nadkarni, Benjamin Kempinski, Anna C. M. Thoeni, Mathieu Lauriere, Maria Polukarov, Pramod Viswanath, Tal Kachman, Yoram Bachrach, Zhangyang 'Atlas' Wang (affiliations include UT Austin / VITA Group, Radboud University, Princeton, and others).

- **Paper Url:** https://neurips.cc/virtual/2025/competition/127731
- **Repo Url:** https://www.mindgamesarena.com/

### Game Coverage

**Games Supported:**

> Four games spanning a spectrum of social/strategic reasoning: Werewolf (adversarial social deduction), Stag Hunt (mixed-motive cooperation), Colonel Blotto (resource-contested competition), and Hanabi (communication-constrained cooperation). Werewolf is the core social-deduction game.

**Hidden Role Mechanic:**

> Hidden roles appear chiefly in the Werewolf track, where a minority of werewolves are secretly assigned among villagers and special roles, and players must infer identities from discussion. The other three games involve hidden intentions/strategies rather than hidden roles.

- **Multimodal Support:** No. Agents communicate via natural language only; no video, audio, or voice-prosody modalities.

### Agent Architecture

**Theory Of Mind Evaluation:**

> Theory of mind is the central evaluation target: the challenge explicitly tests modeling others' beliefs, detecting deception, coordinating under uncertainty, and long-horizon planning. ToM is measured indirectly through head-to-head performance across the four games rather than via a dedicated belief-accuracy probe.


### Evaluation

**Metrics:**

> Performance is measured by head-to-head competition results aggregated into TrueSkill ratings across the four games; per-game win/score outcomes feed the rating.

- **Rating System:** TrueSkill. Agents are rated via TrueSkill computed from head-to-head matches in a live arena.

**Competition Format:**

> A live, head-to-head tournament arena across the four games, with two/three divisions (Open and Efficient agent divisions; Werewolf has Efficient, Unlimited, and Generalization tracks). Cross-play between submitted agents, July-October 2025, $10K prize pool, results presented at NeurIPS 2025 (December 7, 2025).

**Data Source Provenance:**

> Synthetic agent-vs-agent gameplay generated live in the competition arena; no curated human gameplay data.


### Uncertain Fields

- Action Tool Design
- Agent Controllability
- Belief State Representation
- Communication Protocol
- Communication Synchrony
- Credit Assignment
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Framework Used
- Human Baseline Comparison
- Identified Limitations
- Key Results
- Llms Evaluated
- Memory Mechanism
- Num Players
- Open Source
- Optimization Cost
- Optimization Target
- Persuasion Modeling
- Phase Handling
- Planning Approach
- Prompt Optimizer Used
- Reasoning Paradigm
- Reproducibility
- Reward Design
- Rl Algorithm
- Rule Variant Coverage
- Self Play Loop
- Strategy Space Analysis
- Training Paradigm
- Turn Structure

---

## 15. The Traitors: Deception and Trust in Multi-Agent Language Model Simulations

### Basic Info

- **Name:** The Traitors: Deception and Trust in Multi-Agent Language Model Simulations
- **Year:** 2025

**Authors Org:**

> Pedro M. P. Curvo (University of Amsterdam). Presented at the NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models.

- **Paper Url:** https://arxiv.org/abs/2505.12923
- **Repo Url:** https://github.com/pedrocurvo/TheTraitors

### Game Coverage

**Games Supported:**

> A single custom social-deduction environment based on the reality TV show 'The Traitors'. A minority of agents (Traitors) covertly mislead a majority (Faithful), who must infer hidden identities through dialogue and voting. It does not cover Werewolf, Mafia, or Avalon directly, though it is conceptually in the same family.

**Hidden Role Mechanic:**

> Agents are secretly assigned as Traitors or Faithful at game start. Traitors know each other's identities and the full hidden state, while Faithful agents have only their own role and must deduce who the Traitors are from public dialogue. This creates asymmetric information that Traitors exploit.

- **Multimodal Support:** No. The simulation is fully text-based; no video, audio, or voice-prosody modalities are handled.

### Agent Architecture

**Reasoning Paradigm:**

> LLM-based natural-language reasoning. Agents reason over persistent memory and evolving social dynamics to produce accusations, defenses, and votes. The work is grounded in formal frameworks from game theory, behavioral economics, and social cognition rather than using ReAct/MCTS/RL policies.

**Memory Mechanism:**

> Agents maintain persistent memory across the game. The released framework describes an 'advanced memory system' that categorizes information, with each agent holding its own memory, role, and LLM client. History of dialogue and prior interactions accumulates and is fed back into agent prompts via a prompt manager.

**Belief State Representation:**

> Beliefs about which agents are Traitors are represented in natural language and updated as the game proceeds (belief updates under asymmetric information). The framework tracks evolving trust/suspicion toward other agents; there is no explicit probabilistic belief table.

**Action Tool Design:**

> Actions (speak/accuse/defend, build alliances, cast banishment votes, and for Traitors, choose murder targets) are produced as structured natural-language outputs orchestrated by a prompt manager and game engine, not as external callable tool functions.

**Theory Of Mind Evaluation:**

> Yes, theory of mind is central. Agents must model other players' beliefs and likely actions, and the evaluation suite captures trust dynamics and collective inference quality, which serve as ToM measures. The headline finding is explicitly framed around a deceiver-vs-detector ToM asymmetry.


### Game Loop Design

**Turn Structure:**

> Round-based loop with synchronous discussion turns. Within a round agents participate in a shared discussion, conduct private reasoning, then cast banishment votes; Traitors additionally eliminate a Faithful agent. Eliminated agents leave and the loop repeats.

**Phase Handling:**

> Distinct phases are managed by the game engine: a round-table discussion phase, a banishment voting phase, and a Traitor 'murder' phase, with outcome revelation between rounds.

**Communication Protocol:**

> Public natural-language discussion visible to all agents, followed by votes; Traitors also have a private channel/coordination among themselves for choosing murder targets.

**Communication Synchrony:**

> Synchronous, turn-based discussion (agents speak in sequence within a round), with votes cast after the discussion concludes; no asynchronous bidding or scheduler-driven interruption.


### Evaluation

**Metrics:**

> A suite of evaluation metrics capturing deception success, trust dynamics, and collective inference quality, plus game outcomes such as survival duration and Traitor vs Faithful win rates / financial outcomes (prize-pot share).

**Deception Metric Design:**

> Deception is quantified via a custom metric set measuring deception success rate (how often Traitors mislead Faithful), trust dynamics, and vulnerability to others' falsehoods. Measurement focuses on false claims and contradictions and on the influence of deceptive statements on subsequent votes, rather than a formal lie taxonomy.

**Persuasion Modeling:**

> Yes, indirectly. The framework analyzes how effectively agents convince others through discourse and the resulting shifts in trust and voting patterns, treating persuasive influence as part of deception success and trust-dynamics metrics.

**Rating System:**

> No formal competitive rating system (no Elo/TrueSkill). Models are compared via aggregate win rates and the evaluation-metric suite across simulation runs.

**Competition Format:**

> Cross-play between LLM agents: different LLM backends are assigned to agents and compared on deception and detection performance within mixed populations.

**Llms Evaluated:**

> DeepSeek-V3, GPT-4o-mini, and GPT-4o (initial experiments). The framework also supports models via OpenAI, DeepSeek, Together AI, Hugging Face Inference, and local MLX models.

**Human Baseline Comparison:**

> No human baseline. All experiments are LLM-vs-LLM simulations; agents are not compared against human players.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay. All data is generated by autonomous multi-agent LLM simulations; no human gameplay data is used.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. Agents are pretrained LLMs prompted via the framework; no SFT, RL, or preference optimization is performed.

**Strategy Space Analysis:**

> The paper analyzes emergent deceptive behavior, trust formation, and strategic communication dynamics, observing how strategies evolve, though it does not formalize a strategy-space-expansion analysis.


### Optimization

**Prompt Optimizer Used:**

> None. No DSPy or prompt-optimization technique (GEPA, MIPROv2, BootstrapFewShot, SIMBA) is used; prompts are managed by a hand-written prompt manager.

- **Rl Algorithm:** None. No RL or preference-optimization algorithm is applied.

**Reward Design:**

> No training reward. Game outcomes (deception success, survival, prize-pot share, win/loss) are used only as evaluation signals.

- **Optimization Target:** Nothing is optimized; the work evaluates pretrained models as-is.

**Self Play Loop:**

> Multi-agent self-play / cross-play generates the simulation data, but there is no iterative training loop that updates agent parameters.

**Credit Assignment:**

> Not applicable; no learning occurs. Deception influence is attributed statistically through the trust-dynamics and deception-success metrics rather than via training credit assignment.


### Engineering

**Framework Used:**

> A custom Python multi-agent framework (TheTraitors) with components for agents, a game engine (traitors_game.py), a prompt manager, an LLM-client abstraction across providers, and a metrics module. No DSPy or LangChain.

- **Open Source:** Yes. Code is publicly released on GitHub (github.com/pedrocurvo/TheTraitors) under CC-BY 4.0.

**Reproducibility:**

> Moderate to good. The framework is open-sourced with YAML config files and a CLI/Python API for running experiments, but results depend on proprietary API models whose versions drift over time.


### Findings

**Key Results:**

> Initial experiments across DeepSeek-V3, GPT-4o-mini, and GPT-4o reveal a notable asymmetry: advanced models such as GPT-4o demonstrate superior deceptive capabilities yet are disproportionately vulnerable to others' falsehoods. This suggests deception skills may scale faster with model capability than deception-detection skills. Deception, trust formation, and strategic communication emerge naturally in the simulations.

**Deception Emergence:**

> Clear emergent deception. Without explicit instruction to lie, Traitor agents fabricate claims, deflect accusations, and build false alliances; deception, trust, and strategic communication emerge from the multi-agent dynamics under asymmetric information.

**Deceiver Detector Asymmetry:**

> Yes, this is the central finding: LLMs are stronger as deceivers than as deception detectors. More capable models deceive better but are also more vulnerable to being deceived, indicating deception scales faster than detection.


### Uncertain Fields

- Agent Controllability
- Identified Limitations
- Num Players
- Optimization Cost
- Planning Approach
- Rule Variant Coverage

---

## 16. Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games (Eckhaus et al., 2025)

### Basic Info

- **Name:** Time to Talk: LLM Agents for Asynchronous Group Communication in Mafia Games (Eckhaus et al., 2025)
- **Year:** 2025

**Authors Org:**

> Niv Eckhaus, Uri Berger, Gabriel Stanovsky (Hebrew University of Jerusalem; University of Melbourne). Published in Findings of ACL: EMNLP 2025.

- **Paper Url:** https://arxiv.org/abs/2506.05309
- **Repo Url:** https://github.com/niveck/LLMafia

### Game Coverage

- **Games Supported:** Mafia (social-deduction game), played as an online group-chat game with human participants.

**Num Players:**

> 7-12 players per game (average ~7.86); 2-3 Mafia depending on player count. 64 human participants total across 21 games.

**Hidden Role Mechanic:**

> Standard Mafia hidden roles: Mafia members know each other and the Bystanders are hidden. Mafia secretly eliminate a player each night; Bystanders try to identify and vote out the Mafia during the day. The LLM agent plays a seat without other players knowing which participant it is.

**Rule Variant Coverage:**

> Single Mafia rule set (Mafia vs Bystanders) with player count varying 7-12; no systematic coverage of multiple distinct role sets or rule variants.

**Multimodal Support:**

> No. Text-only group chat; the agent adds a simulated typing delay (~1 word/second) but no audio/video modality.


### Agent Architecture

**Reasoning Paradigm:**

> Prompt-based reasoning with a two-module pipeline: a generator that decides what to say and a scheduler that decides whether/when to say it. Adaptive prompting switches between talkative-oriented and listening-oriented instructions. No RL or explicit search.

**Planning Approach:**

> No explicit search or formal plan. The generator composes the next message and the scheduler decides timing; planning is implicit in the per-decision generate/schedule loop run continuously.

**Action Tool Design:**

> Actions are not exposed as formal callable tools. The pipeline has two structured decision points: the scheduler outputs a send/wait decision, and the generator outputs message text (and, in phase, votes / night actions). A custom async game platform applies them.

**Theory Of Mind Evaluation:**

> No dedicated ToM metric. Social blending is measured indirectly via how often human players correctly identify the agent and via post-game human ratings of similarity, timing and relevance.


### Game Loop Design

**Turn Structure:**

> No predefined turns. Daytime is free-form asynchronous discussion and voting; nighttime is Mafia-only elimination. The agent runs continuous decision loops rather than waiting for an assigned turn.

**Phase Handling:**

> Daytime (discussion and voting) and Nighttime (Mafia-only elimination) phases are managed by the custom async game platform; the agent acts within whichever phase is active.

**Communication Protocol:**

> Asynchronous public group chat (Discord-like interface) for day discussion, private Mafia channel at night, and structured voting. Messages have simulated typing delays proportional to length.

**Communication Synchrony:**

> Asynchronous: there is no notion of turns. The core contribution is a scheduler module that decides when to speak (send vs wait), with dynamic prompting tuned to the agent's messaging rate against a 1/n fair-share threshold, modeling realistic group-chat timing.


### Evaluation

**Metrics:**

> Game win rate (Mafia and Bystander); message-timing statistics (frequency, variance); message length; human detection accuracy (how often humans identify the agent); message separability (F1 of LLM-vs-human and Mafia-vs-Bystander classifiers); and post-game human ratings of similarity, timing and relevance.

**Deception Metric Design:**

> No dedicated deception metric. Deception/blending is assessed indirectly through human detection accuracy (59.6%) and classifier separability of agent vs human messages; there is no detector-accuracy surrogate run as a formal deception score or lie taxonomy.

**Persuasion Modeling:**

> Persuasion is not explicitly modeled or optimized. The paper notes a related social effect: the most talkative players are significantly more likely to be voted out, informing the scheduler's restraint, but persuasive influence itself is not optimized.

- **Rating System:** None. No Elo/TrueSkill; performance is reported as win rates and behavioral statistics.

**Competition Format:**

> Mixed human-agent games: a single LLM agent plays alongside multiple human participants in real online Mafia matches (not LLM-vs-LLM self-play, not a model tournament).

**Llms Evaluated:**

> Llama 3.1-8B-Instruct as the agent's base model (both generator and scheduler). Larger models were not tested.

**Human Baseline Comparison:**

> Yes, directly. The agent plays with 64 human participants over 21 games; performance is compared to human players on win rate, message timing and length, and humans serve as the blending-in baseline (they identify the agent only 59.6% of the time).

**Data Source Provenance:**

> LLM-vs-human gameplay: a unique LLMafia dataset of 21 online Mafia games (2,558 messages, 211 from the agent, ~8.2% of messages) with informed-consent human participants who knew an AI was present but not which player it was.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. The agent uses an off-the-shelf Llama 3.1-8B-Instruct with adaptive prompts; no fine-tuning, SFT or RL.

**Agent Controllability:**

> Indirectly. The scheduler's dynamic prompting adjusts talkativeness toward a fair-share 1/n target, effectively controlling participation rate, but there is no tuning to a target win rate or skill level.

**Strategy Space Analysis:**

> Limited. The paper analyzes timing strategy (matching human messaging cadence) and the social cost of verbosity (talkative players get voted out) but does not study emergent-strategy or strategy-space expansion.


### Optimization

- **Prompt Optimizer Used:** None. No DSPy or automated prompt optimizer; the adaptive scheduler prompts are hand-designed.
- **Rl Algorithm:** None. No RL or preference optimization.

**Reward Design:**

> No training reward. The 1/n fair-share threshold acts as a heuristic target for the scheduler's prompt selection, not as a learned reward signal.

**Optimization Target:**

> Nothing is trained; the agent design (generator/scheduler split, dynamic prompting) is the contribution, not an optimized model.

- **Self Play Loop:** No self-play training loop. Games are mixed human-agent matches used for evaluation.

**Credit Assignment:**

> Not applicable in a training sense; there is no learning algorithm assigning credit over multi-turn dialogue or delayed outcomes.


### Engineering

**Framework Used:**

> Custom asynchronous Mafia game platform with a Discord-like chat interface; the agent is a custom two-module (generator + scheduler) LLM pipeline. No DSPy/LangChain.

**Open Source:**

> Yes. Code and the LLMafia dataset are released (GitHub niveck/LLMafia; dataset also on Hugging Face).

**Reproducibility:**

> Good. Code, the LLMafia dataset and a project page are public and the base model (Llama 3.1-8B-Instruct) is named; the main limitation is that live human games cannot be exactly re-run.


### Findings

**Key Results:**

> An adaptive asynchronous LLM agent splitting decision-making into a generator (what to say) and a scheduler (when to say it) plays online Mafia with humans on par with human players in game-performance metrics. The agent blends in well: humans identify it only 59.6% of the time (nearly 40% fail to detect it). Message-timing patterns resemble humans (similar message count with reduced variance), though agent messages are longer (10.67 vs 4.19 words). A separate classifier still distinguishes agent vs human messages (F1=0.98) and Mafia vs Bystander messages (F1=0.88). The most talkative players are significantly more likely to be voted out, validating the scheduler's restraint. Post-game human ratings: 2.63/5 similarity, 3.19/5 timing, 2.99/5 relevance.

**Identified Limitations:**

> Only a small 8B model was tested (larger models untested); the dataset is small (21 games); many participants are non-native English speakers, which may affect how easily agent vs human messages are distinguished; agent messages remain stylistically separable from human ones.

**Deception Emergence:**

> Deception is inherent to the Mafia setting rather than explicitly studied: the agent must blend in and, when Mafia, conceal its role. Evidence of effective concealment is that humans detect the agent only 59.6% of the time, but the paper does not analyze specific emergent deceptive tactics.

**Deceiver Detector Asymmetry:**

> Not explicitly studied. The paper measures human ability to detect the agent (a detection task) but does not run a controlled comparison of LLM strength as deceiver versus as deception detector.


### Uncertain Fields

- Belief State Representation
- Memory Mechanism
- Optimization Cost

---

## 17. Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning

### Basic Info

- **Name:** Training Language Models for Social Deduction with Multi-Agent Reinforcement Learning
- **Year:** 2025

**Authors Org:**

> Bidipta Sarkar, Warren Xia, C. Karen Liu, Dorsa Sadigh (Stanford University). Published at AAMAS 2025.

- **Paper Url:** https://arxiv.org/abs/2502.06060

**Repo Url:**

> https://github.com/SocialDeductionLLM/SocialDeductionLLM (project page: https://socialdeductionllm.github.io/)


### Game Coverage

**Games Supported:**

> An embodied social deduction game based on Among Us, where crewmates must identify an adversarial impostor.

**Hidden Role Mechanic:**

> One hidden adversarial impostor among crewmates; the impostor's identity is unknown private information that crewmates must infer from environment observations and discussion, while the impostor knows its own role.

**Rule Variant Coverage:**

> A single rule configuration — a simplified Among Us-style environment with crewmates versus one impostor; no multiple role sets or rule variants are studied.

**Multimodal Support:**

> No video/audio modalities; the game is an embodied 2D grid environment but agents act and communicate purely through text tokens.


### Agent Architecture

**Reasoning Paradigm:**

> Learned RL policy on a language model. The communication problem is decomposed into a listening sub-task (predicting world information from discussion) and a speaking sub-task (generating influential messages), both optimized via multi-agent RL.

**Memory Mechanism:**

> The agent is initialized from the RWKV-4 World 1.5B model, an RNN-style architecture chosen for unbounded context fine-tuning and constant-time/space generation; game history is carried in the RNN recurrent state rather than an external memory store.

**Belief State Representation:**

> Belief about the impostor's identity is represented as a learned predictive distribution over which player is the impostor; the listening reward trains this belief, and belief change is used as the speaking signal.

**Planning Approach:**

> No explicit planning module or search; the agent acts via a reactive learned RL policy. Strategy emerges from reward shaping rather than deliberate plans.

**Action Tool Design:**

> Actions are emitted as text tokens by the language model — movement and task actions in the embodied environment plus discussion messages and votes during meetings; not exposed as separate tool/function APIs.


### Game Loop Design

**Turn Structure:**

> Embodied task turns (move, complete tasks, observe) alternating with discrete discussion/meeting turns where agents send messages and vote.

**Phase Handling:**

> Two phases managed: a task/exploration phase (agents move and complete tasks while the impostor can eliminate crewmates) and a meeting/discussion phase with multi-round messaging followed by a vote to eject a suspect.

**Communication Protocol:**

> Public natural-language discussion messages exchanged during meeting phases, followed by votes; communication is the channel crewmates use to share evidence about the impostor.

**Communication Synchrony:**

> Synchronous turn-based discussion: agents take ordered speaking turns within meeting rounds rather than asynchronous or bidding-based speech.


### Evaluation

**Metrics:**

> Primary metric is crewmate win rate; secondary metric is belief-prediction accuracy (correctly identifying the impostor). Ablations compare RL-only, listening-only, speaking-only, and combined variants.

**Persuasion Modeling:**

> Yes — the speaking reward explicitly models persuasive influence: a message is rewarded by how much it shifts other agents' beliefs about the impostor's identity, directly optimizing communicative impact.

**Competition Format:**

> Multi-agent self-play / co-training: agents are trained together in the social deduction environment without human demonstrations, and evaluated by crewmate win rate against the adversarial impostor.

**Llms Evaluated:**

> RWKV-4 World 1.5B as the base agent model; the contribution is the training method rather than a broad model comparison.

**Human Baseline Comparison:**

> No human-player baseline; comparisons are against RL-only and ablated training variants, not against humans.

**Data Source Provenance:**

> Fully synthetic LLM-vs-LLM (multi-agent self-play) gameplay generated in the Among Us-based environment; no human demonstrations or curated human gameplay are used.


### Training Methodology

**Training Paradigm:**

> Multi-agent reinforcement learning from scratch with no human demonstrations; the communication problem is decomposed into separately trained listening and speaking objectives.

**Strategy Space Analysis:**

> Yes — the paper analyzes emergent communication behaviors such as accusing suspects and providing supporting evidence that arise without being explicitly programmed.


### Optimization

- **Prompt Optimizer Used:** None — this is a weight-training RL approach, not a DSPy/prompt-optimization approach.

**Reward Design:**

> Dense auxiliary rewards on top of the sparse win/loss outcome: a listening reward trains agents to predict the impostor's identity from discussion, and a speaking reward r^s_t = B_t - B_{t'} rewards each message by how much it improves other agents' beliefs about the impostor.

**Optimization Target:**

> Policy weights of the RWKV language-model agent, jointly optimizing environment actions, listening (belief prediction), and speaking (message influence).

**Self Play Loop:**

> Yes — agents are co-trained via multi-agent interaction (self-play) in the environment, generating their own training data without external demonstrations.

**Credit Assignment:**

> The dense speaking reward provides per-message credit assignment: each utterance is credited by the change it causes in other agents' impostor beliefs (B_t - B_{t'}), converting a sparse delayed game outcome into immediate per-turn signals.


### Engineering

**Framework Used:**

> Custom multi-agent RL training and inference code (released) built on the RWKV-4 World 1.5B model and an Among Us-based embodied environment; not DSPy or LangChain.

**Open Source:**

> Yes — training and inference code and trained models are publicly released via the project site and GitHub.

**Reproducibility:**

> Code and models are released with the project page; moderate reproducibility, though some hyperparameter and algorithm details are sparsely documented.


### Findings

**Key Results:**

> Decomposing communication into listening and speaking with dense rewards roughly doubles crewmate win rates compared to standard RL. Emergent behaviors such as accusing suspects and citing evidence appear without human demonstrations, and combined listening+speaking training outperforms either component alone.

**Deception Emergence:**

> Emergent cooperative-communication behaviors (accusing suspects, providing evidence) appear on the crewmate side; the impostor acts adversarially, but the paper emphasizes truthful evidence-sharing among crewmates rather than analyzing rich emergent impostor deception.


### Uncertain Fields

- Agent Controllability
- Deceiver Detector Asymmetry
- Deception Metric Design
- Identified Limitations
- Num Players
- Optimization Cost
- Rating System
- Rl Algorithm
- Theory Of Mind Evaluation

---

## 18. WOLF: Werewolf-based Observations for LLM Deception and Falsehoods

### Basic Info

- **Name:** WOLF: Werewolf-based Observations for LLM Deception and Falsehoods
- **Year:** 2025

**Authors Org:**

> Mrinal Agarwal, Saad Rana, Theo Sundoro, Hermela Berhe, Spencer Kim, Vasu Sharma, Sean O'Brien, Kevin Zhu (Algoverse AI Research)

- **Paper Url:** https://arxiv.org/abs/2512.09187
- **Repo Url:** https://github.com/MrinalA2009/WOLF-Werewolf-based-Observations-for-LLM-Deception-and-Falsehoods

### Game Coverage

**Games Supported:**

> Werewolf (a single hidden-role social-deduction game), used as a benchmark for measuring deception production and detection.

- **Num Players:** 8 players per game: 4 Villagers, 2 Werewolves, 1 Seer, 1 Doctor (fixed distribution).

**Hidden Role Mechanic:**

> Standard hidden-role Werewolf: Werewolves know each other and conceal their identity; Villagers, Seer and Doctor are hidden. Role-grounded agents receive role-specific objectives; night-action outcomes are revealed without exposing actors. Private scratchpads keep role-specific reasoning from leaking to other agents.

**Rule Variant Coverage:**

> Single fixed 8-player role distribution; no systematic variation of role sets or rule variants (noted as a limitation).

- **Multimodal Support:** No. Text-only social-deduction play.

### Agent Architecture

**Reasoning Paradigm:**

> Chain-of-thought reasoning with private per-agent scratchpads, enabling role-specific internal deliberation without information leakage. No RL policy or explicit search.

**Memory Mechanism:**

> Longitudinal suspicion tracking: each agent accumulates historical judgments via exponential smoothing (alpha = 0.7) over per-statement suspicion scores, weighting recent observations while retaining earlier evidence across debate rounds.

**Belief State Representation:**

> Explicit numeric belief state: suspicion scores in [0,1] per observer-target pair, updated continuously across debate rounds, plus a cross-perception matrix capturing how each agent perceives every other agent's deceptiveness.

**Planning Approach:**

> Role-conditioned strategic objectives rather than explicit search: Werewolves plan to conceal identity and coordinate, Villagers to cooperate transparently, and Seer/Doctor to time information reveals. Planning is implicit in role-grounded prompted reasoning.

**Action Tool Design:**

> Actions are structured decision points in a programmable LangGraph state machine: an integer bidding system (0-10 bids, ties broken by mention priority) sets speaker order, debate utterances are emitted per turn, and a majority vote decides exile; night actions (Werewolf elimination, Doctor protection, Seer investigation) resolve simultaneously.

**Theory Of Mind Evaluation:**

> Yes, central to the work. ToM is operationalized through peer suspicion assessments and a cross-perception matrix: every statement carries self-assessed honesty from the speaker and peer-rated deceptiveness from observers, allowing measurement of how accurately agents model others' deceptiveness.


### Game Loop Design

**Turn Structure:**

> Alternating Night and Day phases. Night: simultaneous Werewolf elimination, Doctor protection and Seer investigation. Day: debate with bidding-determined sequential speaker order followed by a majority-vote exile attempt. Per-game averages: 3.4 nights, 16.1 debate turns, 2.6 voting rounds.

**Phase Handling:**

> A programmable LangGraph state machine enforces strict night-day cycles with explicit phase transitions, rule enforcement and comprehensive event logging (NDJSON).

**Communication Protocol:**

> Public natural-language debate during the day, private night-action channels with outcomes-only revelation, private Werewolf coordination, and majority-rule voting. Every statement is logged as a discrete analysis unit.

**Communication Synchrony:**

> Day debate is sequential with speaker order set by integer bids (0-10), tie-broken by mention priority; night-phase actions resolve simultaneously. It is asynchronous in the sense that turn order is bid-determined rather than fixed.


### Evaluation

**Metrics:**

> Deception Production Rate (share of Werewolf turns containing deceptive statements); detection precision, recall and balanced accuracy; calibration metrics (Brier score, ROC AUC, AUPRC); temporal suspicion dynamics (per-round change in suspicion); and a deception-type taxonomy distribution.

**Deception Metric Design:**

> Deception is measured with a dual signal per statement: a binary self-assessed honesty/deception flag from the speaker and a continuous [0,1] peer-rated deceptiveness score from observers, exponentially smoothed over time. A five-category taxonomy classifies statements as omission, distortion, fabrication, misdirection, or none (truthful). Peer detection accuracy serves as the deception-detectability measure.

**Persuasion Modeling:**

> Not a primary focus; persuasion is not explicitly modeled or optimized. The bidding mechanism lets agents seize the floor, but persuasive influence is not separately quantified.

- **Rating System:** None. No Elo/TrueSkill; results are reported as aggregate rates and detection statistics.

**Competition Format:**

> Self-play simulation: 100 runs of role-grounded LLM agents; asymmetric win conditions (Villagers eliminate all Werewolves; Werewolves win when their count equals/exceeds Villagers'). No cross-model tournament.

**Human Baseline Comparison:**

> No human players. Peer (agent) detection serves as the internal baseline against which deception detectability is measured; there is no human comparison.

- **Data Source Provenance:** Synthetic LLM-vs-LLM gameplay: 7,320+ statements across 100 simulated runs, fully logged in NDJSON.

### Training Methodology

- **Training Paradigm:** Tuning-free: zero-shot role-grounded prompting with role-specific objectives; no fine-tuning or RL.

**Agent Controllability:**

> Not addressed; agent behavior is governed by role-conditioned prompts, with no mechanism to tune to a target win rate or difficulty.

**Strategy Space Analysis:**

> Yes, descriptively. The paper analyzes deception-type dynamics (omission and misdirection persist longer than fabrications) and temporal suspicion trends, showing strategy patterns emerge from role-asymmetric incentives.


### Optimization

- **Prompt Optimizer Used:** None. No DSPy or automated prompt optimizer; prompts are hand-written role-grounded templates.
- **Rl Algorithm:** None. No RL or preference optimization.

**Reward Design:**

> No training reward. Win/loss outcomes and per-statement suspicion are measurement signals, not optimization targets.

**Optimization Target:**

> Nothing is algorithmically optimized; WOLF is an observation/measurement benchmark, not a training method.

**Self Play Loop:**

> 100 self-play runs are executed for measurement, but there is no iterative self-play training loop that updates agents.

**Credit Assignment:**

> Per-statement: each statement is a discrete analysis unit with its own honesty self-label and peer deceptiveness rating, and exponential smoothing (alpha = 0.7) propagates suspicion across turns; this is measurement-level attribution, not learning credit assignment.


### Engineering

**Framework Used:**

> LangGraph-based programmable state machine with strict phase transitions, rule enforcement and NDJSON event logging.

**Open Source:**

> Yes. All code is released on GitHub; full prompts, outputs and NDJSON logs are preserved for audit and replication.

**Reproducibility:**

> High. The state machine, full prompts, raw outputs and NDJSON logs are public, supporting audit and replication; the main gap is the unspecified backing LLM(s).


### Findings

**Key Results:**

> Across 7,320 statements over 100 runs, Werewolves produced deceptive statements in 31% of turns. Peer detection achieved 71-73% precision but only 48-61% recall, yielding about 0.52 balanced accuracy; calibration was underconfident but predictive (Brier 0.26-0.29, ROC AUC ~0.55-0.58, AUPRC ~0.75). Suspicion toward Werewolves rose from ~52% to over 60% across rounds (~+1.6 points per round) while suspicion toward Villagers and the Doctor stayed flat near 44-46%, showing extended interaction improves recall against liars without compounding errors against truthful roles. Werewolves won 70/100 games. Omission and misdirection persist longer than fabrications.

**Identified Limitations:**

> Fixed role distribution and bounded debate length may under-sample longer-horizon strategies; deception labels derive from model self-assessments rather than human annotation; results represent stylized Werewolf play and may not generalize beyond hidden-role social deduction; the backing LLM is not clearly enumerated.

**Deception Emergence:**

> Yes. Deception emerges naturally from role-asymmetric incentives without explicit instruction to be dishonest: Werewolves deceive in 31% of turns, mostly via omission and misdirection (which persist longer) rather than outright fabrication, and self-report a ~69% deception base rate.

**Deceiver Detector Asymmetry:**

> Yes, this is the headline finding: LLMs can deceive convincingly but are weak at detecting deception in peers. High detection precision (~72%) masks low recall (~48%), so subtle lies persist despite widespread suspicion, demonstrating a clear deceiver-stronger-than-detector asymmetry.


### Uncertain Fields

- Llms Evaluated
- Optimization Cost

---

## 19. Who's the Impostor? Multi-Agent Social Deduction for Evaluating LLM Social Reasoning (The Impostor Game)

### Basic Info

**Name:**

> Who's the Impostor? Multi-Agent Social Deduction for Evaluating LLM Social Reasoning (The Impostor Game)

- **Year:** 2025
- **Authors Org:** Xiang Fu. Presented at the NeurIPS 2025 LLM Evaluation Workshop.
- **Paper Url:** https://openreview.net/forum?id=UiUd5LAoq9

### Game Coverage

**Games Supported:**

> A single lightweight custom social-deduction game, the 'Impostor Game' (a word-based majority/impostor deduction game in the spirit of Undercover / Who-is-the-Spy and Among Us). It does not cover Werewolf, Mafia, or Avalon.

- **Num Players:** 4 players per game: 3 agents share a majority word and 1 agent receives a related impostor word.

**Hidden Role Mechanic:**

> Hidden roles arise from asymmetric word knowledge. Three agents are given a common majority word (w_m) and one agent is secretly given a related but different impostor word (w_i). No agent knows the others' words; the impostor must avoid revealing that their word differs while the majority must identify the odd one out from descriptions.

- **Multimodal Support:** No. The game is fully text-based; no video, audio, or voice-prosody modalities.

### Agent Architecture

**Theory Of Mind Evaluation:**

> Theory of mind is evaluated behaviorally. The benchmark measures interactive social reasoning through recognition (detection) accuracy, vote-network influence, and coalition formation, which act as proxies for modeling other agents' knowledge and intent rather than via a dedicated ToM probe.


### Game Loop Design

**Turn Structure:**

> Single-round structure: each of the 4 agents takes a turn to describe its word (speaking order randomized), then all agents cast a vote to identify the impostor.

**Phase Handling:**

> Two phases: a description/speaking phase followed by a voting phase. The game is short, so phase management is simple compared with day/night cycle games.

**Communication Protocol:**

> Public natural-language descriptions visible to all players, followed by a public/aggregated vote. No private channels.

**Communication Synchrony:**

> Synchronous turn-based: agents speak in a randomized order, one at a time, then vote simultaneously. An ITT analysis on speaking position is part of the study.


### Evaluation

**Metrics:**

> Impostor win rate, recognition / detection accuracy, vote-network centrality and coalition efficiency, and speaking-order (seat/position) effects on impostor odds.

**Deception Metric Design:**

> Deception is measured indirectly via the impostor win rate and via detection accuracy (how often the majority correctly identifies the impostor). Influence is quantified through vote-network position rather than a formal lie taxonomy.

**Persuasion Modeling:**

> Yes, influence is explicitly modeled through vote-network analysis: an agent's vote-network centrality strongly predicts realized influence and coalition efficiency (r approximately 0.87), capturing persuasive impact on others' votes.

**Competition Format:**

> Cross-play / large-scale round-robin style evaluation: 9 models are run across 5 modes for a total of 90,720 games, pitting models against each other in mixed compositions.

- **Data Source Provenance:** Synthetic LLM-vs-LLM gameplay generated across 90,720 games; no human gameplay data.

### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. Pretrained LLMs are evaluated as-is across the experimental modes; no SFT or RL.

**Strategy Space Analysis:**

> The study analyzes emergent influence and coalition dynamics, including the counterintuitive result that team-aware agents underperform team-unaware ones; it does not perform a formal strategy-space-expansion analysis.


### Optimization

- **Prompt Optimizer Used:** None. No DSPy or prompt optimizer (GEPA, MIPROv2, BootstrapFewShot, SIMBA) is used.
- **Rl Algorithm:** None. No RL or preference-optimization algorithm is applied.
- **Reward Design:** No training reward. Win/detection outcomes are used only as evaluation signals.
- **Optimization Target:** Nothing is optimized; the work evaluates pretrained models.

**Self Play Loop:**

> Large-scale self-play / cross-play generates evaluation data, but there is no iterative training loop updating the agents.

**Credit Assignment:**

> Not applicable; no learning. Influence is attributed via vote-network analysis rather than training credit assignment.


### Findings

**Key Results:**

> Across 90,720 games (9 models, 5 modes), vote-network position best explains realized influence: coalition efficiency rises with centrality (r approximately 0.87). Impostor win rate varies substantially across models (27.8%-69.0%). Recognition/detection accuracy shows a positive cross-model trend with outcomes (r approximately 0.61, p=0.17). A pseudo-arm ITT analysis indicates middle/late speaking modestly reduces impostor odds (about -1.11 percentage points; 95% CI [-1.64, -0.55]). Counterintuitively, agents given team-awareness information underperformed those without it.

**Deception Emergence:**

> Impostors must implicitly deceive by describing a different word convincingly; impostor win rates of 27.8-69.0% show varying success at evading detection, evidencing emergent deceptive behavior, though it is not analyzed via a lie taxonomy.


### Uncertain Fields

- Action Tool Design
- Agent Controllability
- Belief State Representation
- Deceiver Detector Asymmetry
- Framework Used
- Human Baseline Comparison
- Identified Limitations
- Llms Evaluated
- Memory Mechanism
- Open Source
- Optimization Cost
- Planning Approach
- Rating System
- Reasoning Paradigm
- Repo Url
- Reproducibility
- Rule Variant Coverage

---

## 20. Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games (Kao et al., 2026)

### Basic Info

**Name:**

> Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games (Kao et al., 2026)

- **Year:** 2026

**Authors Org:**

> Christopher Kao, Vanshika Vats, James Davis (University of California, Santa Cruz). Accepted to IEEE ICA 2025.

- **Paper Url:** https://arxiv.org/abs/2601.13709
- **Repo Url:** https://github.com/cocochief4/llm-mafia

### Game Coverage

- **Games Supported:** Mafia (social-deduction game), used to study deception quality.

**Hidden Role Mechanic:**

> Standard Mafia hidden roles: Mafia members know each other and conceal their identity; Bystanders/Villagers are hidden and try to deduce who the Mafia are through conversation. Success for the Mafia depends on deceiving others.

- **Multimodal Support:** No. Text-only Mafia gameplay and transcript analysis.

### Agent Architecture

**Reasoning Paradigm:**

> Prompt-based LLM agents play Mafia (no RL or explicit search); a separate prompt-based 'Mafia Detector' LLM analyzes transcripts to predict Mafia players. The architecture is a transcript-generation step plus a detection/analysis step.

**Theory Of Mind Evaluation:**

> No dedicated ToM metric. ToM is implicit in the deception-quality measurement: the Mafia Detector's accuracy at inferring hidden roles from transcripts serves as a proxy for how well agents conceal versus reveal intent.


### Game Loop Design

**Communication Protocol:**

> Public natural-language discussion and voting during the day with private Mafia coordination at night, conducted in an asynchronous multi-agent framework rather than strict turn-taking.

**Communication Synchrony:**

> Asynchronous. The paper explicitly adopts an asynchronous multi-agent framework, arguing it better simulates realistic social contexts than the synchronous turn-based setups used in prior Mafia/SDG work.


### Evaluation

**Metrics:**

> Mafia Detector prediction accuracy (probability the detector correctly identifies Mafia players) used as the primary surrogate for deception quality, compared across LLM games, human games and a random baseline; results are reported across game days and across the number of Mafia detected.

**Deception Metric Design:**

> Detector-accuracy surrogate: a GPT-4-Turbo 'Mafia Detector' analyzes game transcripts with all player-role information removed and predicts who the Mafia are. Lower detector accuracy indicates higher deception quality (better blending in). This is benchmarked against detector accuracy on human transcripts and a random baseline.

**Persuasion Modeling:**

> Persuasion is not explicitly modeled or optimized; the focus is on concealment/blending-in quality rather than persuasive influence.

- **Rating System:** None. No Elo/TrueSkill; results are reported as detector accuracy comparisons.

**Competition Format:**

> LLM self-play games (35 simulated Mafia games with LLM agents) compared against a separate corpus of 28 human games; not a head-to-head model tournament.

**Llms Evaluated:**

> GPT-4o as the gameplay agent for all 35 LLM Mafia games; GPT-4-Turbo as the Mafia Detector that analyzes transcripts.

**Human Baseline Comparison:**

> Yes, central to the work. Detector accuracy on 35 LLM games is directly compared against 28 human Mafia games and a random baseline; the comparison shows the detector is less accurate on LLM games, meaning LLMs deceive/blend in more effectively than humans.

**Data Source Provenance:**

> Mixed. Synthetic LLM-vs-LLM Mafia transcripts (35 games of GPT-4o agents) plus a curated set of 28 real human Mafia games used as the baseline; the LLM transcript dataset is released.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting. Both the GPT-4o players and the GPT-4-Turbo detector are off-the-shelf prompted models; no fine-tuning or RL.

**Agent Controllability:**

> Not addressed; agents are fixed prompted GPT-4o instances with no mechanism to tune skill or win rate.

**Strategy Space Analysis:**

> Limited. The paper analyzes deception quality across game days and number of Mafia but does not study emergent-strategy taxonomies or strategy-space expansion.


### Optimization

- **Prompt Optimizer Used:** None. No DSPy or automated prompt optimizer; prompts are hand-written.
- **Rl Algorithm:** None. No RL or preference optimization.

**Reward Design:**

> No training reward. Detector prediction accuracy is an evaluation/surrogate measure, not an optimization signal.

**Optimization Target:**

> Nothing is algorithmically optimized; the contribution is a measurement methodology and dataset, not a training method.

**Self Play Loop:**

> 35 LLM self-play games are run for measurement, but there is no iterative self-play training loop that updates the agents.

**Credit Assignment:**

> Not applicable in a training sense; the detector attributes Mafia predictions per player over a whole transcript, but there is no learning credit assignment over turns.


### Engineering

**Framework Used:**

> Custom asynchronous multi-agent Mafia framework plus a separate GPT-4-Turbo-based Mafia Detector for transcript analysis. No DSPy/LangChain reported.

**Open Source:**

> Partially. A dataset of LLM Mafia transcripts is released (GitHub cocochief4/llm-mafia); full code release status for the framework is not clearly confirmed.

**Reproducibility:**

> Moderate. The LLM transcript dataset is public and the models (GPT-4o, GPT-4-Turbo) and methodology are named, but the games rely on closed API models and the human-game corpus and full code availability are less clearly specified.


### Findings

**Key Results:**

> Using an asynchronous Mafia framework and a GPT-4-Turbo Mafia Detector that predicts Mafia from role-stripped transcripts, the study finds the detector's accuracy is lower on LLM (GPT-4o) games than on human games, and this holds consistently across game days and number of Mafia detected. The conclusion is that LLMs blend in better than humans and therefore deceive more effectively. The authors release a dataset of LLM Mafia transcripts.

**Identified Limitations:**

> Only GPT-4o was used for gameplay and GPT-4-Turbo for detection; small sample (35 LLM and 28 human games); detailed per-game player counts and architectural choices are not specified; the detector is itself an imperfect LLM proxy for deception quality.

**Deception Emergence:**

> Yes. LLM agents produce transcripts in which an LLM detector struggles to identify the Mafia, indicating emergent effective deception/concealment; notably the LLMs blend in better than human players, deceiving more effectively in the asynchronous setting.

**Deceiver Detector Asymmetry:**

> The work centers on deception production rather than a controlled deceiver-vs-detector comparison, but its core finding (an LLM detector fails to catch LLM Mafia) implies LLMs are stronger as deceivers than the LLM-based detector is at catching them.


### Uncertain Fields

- Action Tool Design
- Belief State Representation
- Memory Mechanism
- Num Players
- Optimization Cost
- Phase Handling
- Planning Approach
- Rule Variant Coverage
- Turn Structure

---

## 21. AmongAgents — Evaluating Large Language Models in the Interactive Text-Based Social Deduction Game (Among Us)

### Basic Info

**Name:**

> AmongAgents — Evaluating Large Language Models in the Interactive Text-Based Social Deduction Game (Among Us)

- **Year:** 2024 (arXiv July 2024; presented at the Wordplay workshop, ACL 2024)
- **Authors Org:** Yizhou Chi (UC Berkeley), Linjiang Mao (Tongji University), Zineng Tang (UC Berkeley).
- **Paper Url:** https://arxiv.org/abs/2407.16521
- **Repo Url:** https://github.com/cyzus/among-agents

### Game Coverage

- **Games Supported:** A text-based adaptation of Among Us, a hidden-role social deduction game.
- **Num Players:** 5 players by default (4 Crewmates + 1 Impostor); configurable.

**Hidden Role Mechanic:**

> One hidden Impostor among Crewmates. The Impostor knows its role and can secretly kill and use vents; Crewmates do not know who the Impostor is and must infer identity from observations and discussion — asymmetric private information.

**Rule Variant Coverage:**

> Primarily the standard 4-Crewmate/1-Impostor Among Us configuration (configurable player count); not a broad multi-variant rule space.

- **Multimodal Support:** Text-only; the original spatial Among Us game is reduced to a text environment with no audio/video.

### Agent Architecture

**Reasoning Paradigm:**

> Prompted LLM agents (instruction-tuned, no RL): each agent processes observations and memory to produce an action probability distribution. Agents use ReAct-style reasoning over observations, memory, and an optional planner.

**Memory Mechanism:**

> Each agent maintains a memory of historical state-action pairs with summarization functions to preserve long-term coherence; agents can be queried about who they have seen, where, and what actions occurred.

**Belief State Representation:**

> Beliefs about the hidden Impostor are represented in natural language within the agent's reasoning and memory; no explicit probability table — suspicions are expressed in discussion and reasoning traces.

**Planning Approach:**

> An optional Planning Module lets agents form and retrieve plans (e.g., task prioritization, elimination targets); ablations show agents without the planner achieve diminished task completion.

**Action Tool Design:**

> Game actions are exposed as a discrete action set: shared actions (move, speak, vote, report), Crewmate-specific (complete tasks), and Impostor-specific (kill, vent); the LLM selects actions each turn within the text environment.

**Theory Of Mind Evaluation:**

> The work explores Theory-of-Mind collaborative capabilities — agents simulate other players' behaviors and intentions — and includes a Reasoning evaluation category for deception detection and behavioral analysis, though there is no single quantitative ToM score.


### Game Loop Design

**Turn Structure:**

> Turn-based: during the Task Phase agents take movement/task/observation turns; during the Meeting Phase agents take ordered discussion turns over multiple rounds followed by a vote.

**Phase Handling:**

> Two phases managed by the environment: Task Phase (movement, task completion, observation, Impostor kills/vents) and Meeting Phase (3 rounds of discussion plus a vote to eject a player).

**Communication Protocol:**

> Public natural-language discussion during Meeting Phases (3 discussion rounds) followed by voting; reporting a body or calling a meeting triggers the discussion channel.

**Communication Synchrony:**

> Synchronous turn-based: discussion proceeds in ordered rounds and voting is collective, not asynchronous or bidding-driven.


### Evaluation

**Metrics:**

> Crewmate task-completion win rate, game outcomes by faction, fraction of utterances containing deception, and performance across the five controlled evaluation categories (self-awareness, memory, planning, reasoning, reflection).

**Deception Metric Design:**

> Deception is quantified by the proportion of utterances containing deceptive content — Impostors used deception in over 40% of utterances while Crewmates primarily used truth-telling — measured via GPT-4-assisted annotation/analysis of dialogue.

**Competition Format:**

> Self-play / cross-play among LLM agents within the environment, with controlled evaluation runs; personality archetypes are pitted against each other to study strategic effectiveness.

**Llms Evaluated:**

> GPT-3.5-turbo as the primary agent model; GPT-4 used for annotation and analysis. The framework is open for evaluating other LLM agents.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay generated within the AmongAgents text environment; no curated real human gameplay.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting — instruction-tuned LLM inference with prompt engineering; no SFT or RL fine-tuning.

**Agent Controllability:**

> Agent behavior is varied through 10 personality archetypes (5 Crewmate, 5 Impostor) selected via posterior win probability, which produce differential strategic effectiveness; this is personality-based variation rather than precise win-rate targeting.

**Strategy Space Analysis:**

> Yes — the paper analyzes how different personality archetypes produce different strategies and outcomes (e.g., 'The Leader' persona correlates with higher casualty risk) and studies emergent deception and collaboration patterns.


### Optimization

- **Prompt Optimizer Used:** None — no DSPy/automated prompt optimizer; hand-crafted prompt-engineering framework.
- **Rl Algorithm:** None — no RL or preference optimization is used.

**Reward Design:**

> No training reward; game outcomes and category scores are used only for evaluation, not for optimization.

- **Optimization Target:** Not applicable — nothing is trained; the framework evaluates prompted LLM agents as-is.

**Self Play Loop:**

> Agents play repeated multi-agent games (used to estimate personality posterior win probabilities), but there is no parameter-updating self-play training loop.

- **Optimization Cost:** Not applicable — no training; cost is limited to LLM API inference for gameplay and GPT-4 analysis.

### Engineering

**Framework Used:**

> Custom prompt-engineering agent framework and a text-based Among Us environment built by the authors; instruction-tuned LLM inference without RL. Not DSPy-based.

**Open Source:**

> Yes — the environment and agent framework code are publicly released at https://github.com/cyzus/among-agents.

**Reproducibility:**

> Good — an open text environment and agent framework with defined evaluation categories; reproducibility depends on LLM API access and configuration of personalities/phases.


### Findings

**Key Results:**

> LLM agents comprehend and follow Among Us rules well: agents equipped with the planner reach a 50% Crewmate task-completion victory rate versus 0% for a random baseline. Impostors employ deception in over 40% of utterances while Crewmates mostly tell the truth. Different personality archetypes show distinct strategic effectiveness. Overall, LLMs follow rules robustly but use suboptimal deception strategies and need better deceptive reasoning.

**Identified Limitations:**

> Agent deception tactics remain suboptimal compared to humans; personality-construction methods and their orthogonality to action distributions need further refinement; the environment is a simplified text reduction of spatial Among Us.

**Deception Emergence:**

> Deceptive behavior emerges from Impostor agents — over 40% of Impostor utterances contain deception (misdirection, false alibis, blame-shifting) — while Crewmates predominantly use truthful communication; deception quality is still below human level.


### Uncertain Fields

- Credit Assignment
- Deceiver Detector Asymmetry
- Human Baseline Comparison
- Persuasion Modeling
- Rating System

---

## 22. AvalonBench / Avalon's Game of Thoughts (AGoT) — Recursive Contemplation (ReCon) for The Resistance: Avalon

### Basic Info

**Name:**

> AvalonBench / Avalon's Game of Thoughts (AGoT) — Recursive Contemplation (ReCon) for The Resistance: Avalon

- **Year:** 2023 (both AvalonBench and Avalon's Game of Thoughts/ReCon; ReCon published at ACL 2024 Findings)

**Authors Org:**

> AvalonBench: Jonathan Light, Min Cai, Sheng Shen, Ziniu Hu (arXiv:2310.05036). Avalon's Game of Thoughts / Recursive Contemplation (ReCon): Shenzhi Wang, Chang Liu, Zilong Zheng, Siyuan Qi, Shuo Chen, Qisen Yang, Andrew Zhao, Chaofei Wang, Shiji Song, Gao Huang (arXiv:2310.01320).

**Paper Url:**

> AvalonBench: https://arxiv.org/abs/2310.05036 ; Avalon's Game of Thoughts (ReCon): https://arxiv.org/abs/2310.01320

**Repo Url:**

> AvalonBench: https://github.com/jonathanmli/Avalon-LLM ; ReCon project page: https://shenzhi-wang.github.io/avalon_recon/


### Game Coverage

**Games Supported:**

> The Resistance: Avalon, a hidden-role social deduction game. AvalonBench provides the game environment; Avalon's Game of Thoughts uses Avalon as a deception testbed.

**Num Players:**

> 5-6 player Avalon games (commonly the 6-player setup with Merlin, Percival, loyal servants, Morgana, Assassin, Mordred-style role configurations).

**Hidden Role Mechanic:**

> Asymmetric hidden roles split into Good (loyal servants of Arthur, plus Merlin/Percival with partial information) and Evil (minions of Mordred, including Assassin and Morgana). Evil players know each other; Merlin knows Evil; Percival sees Merlin/Morgana ambiguously — creating layered information asymmetry.

**Rule Variant Coverage:**

> Primarily the standard Avalon role set with special roles (Merlin, Percival, Morgana, Assassin); limited explicit enumeration of multiple distinct rule variants — coverage centers on one canonical configuration.

- **Multimodal Support:** Text-only; no video, audio, or voice-prosody modalities.

### Agent Architecture

**Reasoning Paradigm:**

> AvalonBench: prompted LLM agents with rule-based action validity. Avalon's Game of Thoughts contributes Recursive Contemplation (ReCon): a recursive-reasoning paradigm with formulation and refinement contemplation, incorporating first-order and second-order perspective transitions (vs. CoT baseline).

**Memory Mechanism:**

> Game history is supplied to agents through the prompt context as accumulated discussion, quest outcomes, and voting records; ReCon adds explicit contemplation steps over this history rather than an external retrieval store.

**Belief State Representation:**

> Beliefs about hidden roles are represented in natural language. ReCon's first-order perspective transition infers others' mental states and the second-order transition models how others perceive the agent's own mental state, all expressed as NL contemplation rather than a probability table.

**Planning Approach:**

> AvalonBench uses prompt-driven action selection within valid game moves. ReCon plans speech and actions through a two-stage contemplation process — formulation contemplation produces initial thoughts/speech and refinement contemplation polishes them — rather than explicit tree search.

**Action Tool Design:**

> Game actions (team proposal, quest voting, quest success/fail cards, Assassin's Merlin guess, discussion speech) are exposed via the AvalonBench environment as a constrained discrete action interface; the LLM selects from valid moves each phase.

**Theory Of Mind Evaluation:**

> Yes — ReCon explicitly targets theory of mind via first-order (inferring others' mental states) and second-order (modeling others' perception of oneself) perspective transitions, evaluated through deception-detection and privacy-preservation outcomes.


### Game Loop Design

**Turn Structure:**

> Round-based: each round involves a leader proposing a team, public discussion, team-approval voting, and (if approved) quest cards; turns proceed in fixed player order.

**Phase Handling:**

> Distinct phases managed by the environment: team-proposal/discussion phase, team-approval voting phase, quest-execution phase, and a final Assassin phase where Evil may identify Merlin.

**Communication Protocol:**

> Public natural-language discussion plus structured votes (team approval) and hidden quest cards; some role knowledge is private. AvalonBench supports discussion among agents during proposal phases.

**Communication Synchrony:**

> Synchronous turn-based discussion and voting; agents speak and vote in fixed order rather than via asynchronous bidding or scheduler-driven speech.


### Evaluation

**Metrics:**

> Win rate by team (Good vs Evil), quest success/fail outcomes; ReCon additionally reports deception-detection success and privacy-preservation (Merlin avoiding identification), with good-side win rate as the headline metric.

**Deception Metric Design:**

> Deception is measured via outcomes: ReCon evaluates whether Evil players can deceive (cause quest failures / mislead) and whether Good players detect deception, plus a privacy-preservation metric for whether Merlin's identity stays hidden from the Assassin.

**Persuasion Modeling:**

> Persuasion is present implicitly through discussion that sways team-approval votes, but it is not formalized as an explicit optimized persuasion objective.

**Competition Format:**

> Cross-play and self-play: LLM agents play against other LLM agents and against rule-based bots in AvalonBench; ReCon compares ReCon-equipped agents against CoT baseline agents within the same games.

**Llms Evaluated:**

> GPT-3.5 and GPT-4 are the primary models evaluated in both AvalonBench and Avalon's Game of Thoughts experiments.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM (and LLM-vs-rule-based-bot) gameplay generated within the AvalonBench environment; not curated real human gameplay.


### Training Methodology

**Training Paradigm:**

> Tuning-free prompting — both AvalonBench agents and ReCon operate via prompting and recursive contemplation with no fine-tuning, SFT, or RL.

**Strategy Space Analysis:**

> Strategies such as deception by Evil and deduction/privacy-preservation by Good are analyzed qualitatively; ReCon shows recursive contemplation expands effective strategic behavior over CoT.


### Optimization

- **Prompt Optimizer Used:** None — no DSPy/automated prompt optimizer; ReCon is a hand-designed recursive prompting framework.
- **Rl Algorithm:** None — no RL or preference optimization is used.

**Reward Design:**

> No training reward; performance is driven by prompt/contemplation design. Game outcome (win/loss, quest success) serves only as an evaluation signal.

**Optimization Target:**

> Reasoning structure / prompts: ReCon optimizes the agent's contemplation procedure (formulation and refinement with perspective transitions) rather than weights or demonstrations.

**Self Play Loop:**

> Agents play repeated games against each other, but there is no iterative self-play training loop that updates model parameters.


### Engineering

**Framework Used:**

> AvalonBench is a custom Avalon game environment (with rule-based bots); ReCon is a custom recursive-prompting framework layered on LLM APIs. Not built on DSPy.

**Open Source:**

> Yes — AvalonBench code/environment is released on GitHub (jonathanmli/Avalon-LLM); ReCon code is available via its project page.

**Reproducibility:**

> Moderate-to-good — AvalonBench provides an open environment with rule-based opponents; reproducibility depends on LLM API access and prompt configurations.


### Findings

**Key Results:**

> AvalonBench shows LLM agents (GPT-3.5/GPT-4) struggle against rule-based bots, especially on the Good side, revealing weak strategic deduction. Avalon's Game of Thoughts shows Recursive Contemplation (ReCon) dramatically improves play: good-side win rate rises from ~15% with a CoT baseline to ~83.3% with ReCon, with markedly better deception detection and privacy preservation.

**Identified Limitations:**

> LLM agents are weak at strategic discussion and deduction (notably on the Good side in AvalonBench); ReCon adds extra LLM calls and is still imperfect at fully resolving deception in complex multi-agent settings.

**Deception Emergence:**

> Evil agents engage in deceptive discussion to mislead Good players and cause quest failures; ReCon's recursive perspective-taking is shown to both improve deceptive play and counter opponents' deception (better detection and identity concealment for Merlin).

**Deceiver Detector Asymmetry:**

> AvalonBench results suggest LLMs are relatively weak as detectors (Good side underperforms), implying an asymmetry favoring deception over detection; ReCon narrows this gap by strengthening detection and privacy preservation.


### Uncertain Fields

- Agent Controllability
- Credit Assignment
- Human Baseline Comparison
- Optimization Cost
- Rating System

---

## 23. DSPy agent stack — ReAct & memory (mem0) tutorials, optimizers (MIPROv2 / GEPA / BootstrapFewShot)

### Basic Info

- **Name:** DSPy agent stack — ReAct & memory (mem0) tutorials, optimizers (MIPROv2 / GEPA / BootstrapFewShot)
- **Year:** DSPy v2 2023; DSPy 2.x with ReAct, MIPROv2, GEPA 2024-2025; mem0 ReAct tutorial 2024-2025

**Authors Org:**

> DSPy: Stanford NLP (Omar Khattab et al.) and open-source community. GEPA: Lakshya A. Agrawal et al. (UC Berkeley / Stanford / Databricks collaboration). MIPROv2: DSPy team (Krista Opsahl-Ong et al.). mem0: Mem0 (Taranjeet Singh et al.).

**Paper Url:**

> DSPy: https://arxiv.org/abs/2310.03714 ; MIPROv2: https://arxiv.org/abs/2406.11695 ; GEPA: https://arxiv.org/abs/2507.19457 ; ReAct: https://arxiv.org/abs/2210.03629 ; mem0: https://arxiv.org/abs/2504.19413

**Repo Url:**

> DSPy: https://github.com/stanfordnlp/dspy ; mem0: https://github.com/mem0ai/mem0 ; DSPy mem0 ReAct tutorial: https://dspy.ai/tutorials/mem0_react_agent/


### Game Coverage

**Games Supported:**

> Not a game library — DSPy is a general LM-program framework. It is game-agnostic and can be wired to any social-deduction environment (Werewolf, Mafia, Secret Hitler, Avalon) by defining signatures and tools; no games ship with it.

**Hidden Role Mechanic:**

> Not provided by DSPy itself. Hidden-role state would be passed into a DSPy module as input fields (e.g., a private 'role' field) and tracked in the agent's memory/observations; the framework imposes no role structure.


### Agent Architecture

**Reasoning Paradigm:**

> ReAct (Reason + Act interleaving) is the core agent module: dspy.ReAct alternates Thought/Action/Observation steps with tool calls. DSPy also offers ChainOfThought, ProgramOfThought, and composition of modules; reasoning is realized as declarative signatures rather than hand-written prompts.

**Memory Mechanism:**

> DSPy itself is largely stateless per call; persistent memory is added as a tool. The official mem0 ReAct tutorial wraps Mem0 in a MemoryTools class exposing five operations — store_memory(), search_memories(), get_all_memories(), update_memory(), delete_memory() — backed by Mem0's vector store. Memory is stored via memory.add() and retrieved via semantic memory.search(); the MemoryReActAgent exposes seven tools total (the five memory ops plus get_current_time and set_reminder/preferences). Game history is thus held in an external semantic store and retrieved on demand, rather than in a fixed memory-stream window.

**Planning Approach:**

> Planning is implicit in ReAct's interleaved Thought steps; DSPy has no dedicated planner module. Explicit planning can be built by composing a planning signature/module that emits a plan consumed by downstream modules, or by exposing a 'planning' tool to ReAct. Search-based planning is not built in.

**Action Tool Design:**

> Game actions are exposed to dspy.ReAct as Python callables passed in the tools list. Each tool is a function with a docstring and typed signature; DSPy auto-generates the tool schema from the function signature and lets the LM choose which tool to call with what arguments. Social-game actions (vote, accuse, claim_role, nominate, kill) would each be a tool function; memory and observation accessors are likewise tools.

**Theory Of Mind Evaluation:**

> Not measured by DSPy. ToM-style reasoning about other players can be implemented as signature fields or memory content and optimized with an optimizer, but DSPy provides no ToM metric — evaluation depends on the user-supplied metric function.


### Game Loop Design

**Turn Structure:**

> Not imposed by DSPy. Within a single agent call, ReAct runs up to max_iters Thought/Action/Observation cycles (e.g., 6 in the mem0 tutorial). Game-level turn order is defined by the surrounding game loop the developer writes.

**Phase Handling:**

> Not handled by DSPy — phase logic (day/night, discussion, voting) lives in the developer's game loop. DSPy modules can be specialized per phase (e.g., a discussion signature vs. a voting signature) and selected by the loop.

**Communication Protocol:**

> Not prescribed. DSPy modules consume and produce text fields; the developer defines whether output is a public message, a private message, or a vote. Inter-agent communication is mediated by the host environment passing transcripts as input fields.


### Evaluation

**Metrics:**

> DSPy uses a user-defined metric function (any callable returning a score/bool) to drive optimization and evaluation; for games this could be win rate, survival, vote accuracy, or speech-quality judgments. No fixed metric is built in.

**Persuasion Modeling:**

> Not modeled by DSPy. Persuasive influence could be made an optimization target via a custom metric (e.g., reward for swinging votes) but is not a built-in concept.

**Competition Format:**

> Not applicable — DSPy is a single-agent program framework; self-play / cross-play / tournament structure is implemented by the developer around DSPy modules.

**Llms Evaluated:**

> Model-agnostic via LiteLLM — supports OpenAI (GPT-4o/o-series), Anthropic Claude, Google Gemini, open models (Llama, Qwen, DeepSeek), and local servers (vLLM, Ollama). The mem0 tutorial uses gpt-4o-mini; optimizer papers benchmark GPT-3.5/4-class and open models.

**Human Baseline Comparison:**

> Not applicable at the framework level — DSPy provides no human baseline; any such comparison is up to the experiment design.


### Training Methodology

**Training Paradigm:**

> Primarily tuning-free prompt optimization (instruction + few-shot search) rather than weight training. DSPy also supports weight finetuning via BootstrapFinetune (SFT) and integrates RL-style optimization (e.g., dspy.GRPO / dspy.ART) for fine-tuning LM weights; the dominant mode is prompt/demonstration optimization.


### Optimization

**Prompt Optimizer Used:**

> DSPy ships several: BootstrapFewShot and BootstrapFewShotWithRandomSearch (bootstrapped few-shot demos), MIPROv2 (joint instruction + few-shot Bayesian optimization), GEPA (reflective evolutionary prompt optimization using natural-language feedback, highly sample-efficient), SIMBA, COPRO, and BootstrapFinetune (weight tuning). GEPA and MIPROv2 are the recommended instruction optimizers.

**Reward Design:**

> Reward = the user-defined metric function output. GEPA additionally consumes rich natural-language feedback traces (not just a scalar) as its optimization signal; MIPROv2/Bootstrap use the scalar metric. For games the reward would be win/loss outcome or auxiliary speech/vote metrics chosen by the developer.

**Optimization Target:**

> Instructions/prompt text and few-shot demonstrations for each predictor (MIPROv2, GEPA, BootstrapFewShot); optionally LM weights (BootstrapFinetune, GRPO). The program structure (signatures, module composition) is author-defined and not optimized.

**Optimization Cost:**

> Prompt optimizers are designed to be sample-efficient: BootstrapFewShot is cheap (tens of LM calls); MIPROv2 and GEPA run more rollouts but GEPA is reported to be highly sample-efficient, often beating MIPROv2 and human prompts with fewer evaluations. Cost scales with trainset size, number of predictors and rollouts.


### Engineering

**Framework Used:**

> DSPy (declarative LM-programming framework, Python) layered on LiteLLM for model access; mem0 used as the external memory backend in the ReAct memory tutorial.

**Open Source:**

> Yes — DSPy is MIT-licensed and open source; mem0 is open source (Apache 2.0). Tutorials, optimizer code, and docs are public on GitHub and dspy.ai.

**Reproducibility:**

> Good — DSPy provides save/load of optimized programs, deterministic-ish runs with caching, public tutorials (incl. the mem0 ReAct tutorial), and documented optimizer APIs; LM nondeterminism and API drift are the main reproducibility risks.


### Findings

**Key Results:**

> DSPy popularized declarative LM programming where optimizers compile signatures into high-performing prompts. The mem0 ReAct tutorial shows a working memory-augmented agent exposing seven tools (five memory ops) with up to six ReAct iterations. MIPROv2 jointly optimizes instructions and demonstrations and consistently beats hand-written prompts; GEPA, using reflective natural-language feedback and evolutionary search, often outperforms MIPROv2 and the best human prompts while being markedly more sample-efficient. The stack — ReAct + mem0 memory tool + GEPA/MIPROv2 optimization — is the recommended methodology for building optimizable tool-using agents.

**Identified Limitations:**

> No native game loop, phase handling, belief tracking, rating system, or self-play — all must be built by the user. Optimization quality depends on a well-designed metric; multi-turn delayed-reward credit assignment for games is not natively supported. LM nondeterminism and API changes can hurt reproducibility. Memory tutorial is illustrative (gpt-4o-mini, simple QA) rather than a social-game benchmark.


### Uncertain Fields

- Agent Controllability
- Belief State Representation
- Communication Synchrony
- Credit Assignment
- Data Source Provenance
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Multimodal Support
- Num Players
- Rating System
- Rl Algorithm
- Rule Variant Coverage
- Self Play Loop
- Strategy Space Analysis

---

## 24. InMind / InMind-Avalon — Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles

### Basic Info

- **Name:** InMind / InMind-Avalon — Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles
- **Year:** 2025 (arXiv August 2025; published at EMNLP 2025 main conference)

**Authors Org:**

> Zizhen Li (Nankai University, Shanghai Innovation Institute) and collaborators from Shanghai AI Laboratory, Fudan University, Johns Hopkins University, Wuhan University, and University of Science and Technology of China.

- **Paper Url:** https://arxiv.org/abs/2508.16072
- **Repo Url:** https://github.com/leroy9472/InMind

### Game Coverage

**Games Supported:**

> The Resistance: Avalon (case study). InMind is a general cognitively grounded evaluation framework for social deduction games, instantiated as InMind-Avalon.

- **Num Players:** 6-player Avalon games.

**Hidden Role Mechanic:**

> Standard Avalon hidden-role structure with Good and Evil factions and special roles holding asymmetric private information; InMind focuses less on the mechanic itself and more on individualized reasoning under this hidden-role uncertainty.

**Rule Variant Coverage:**

> A single canonical 6-player Avalon configuration; the contribution is reasoning-style evaluation rather than rule-variant breadth.

**Multimodal Support:**

> Text-only; gameplay transcripts, strategy traces, and reflections are all textual. No audio/video modalities.


### Agent Architecture

**Reasoning Paradigm:**

> Evaluation framework rather than a single agent paradigm — it probes LLMs (general-purpose and reasoning-enhanced models) on capturing and applying individual human reasoning styles. Agents are evaluated via prompting on four cognitive tasks.

**Memory Mechanism:**

> The InMind-Avalon dataset enriches structured gameplay with round-level strategy traces and post-game reflective summaries; models are given this history as context to test temporal alignment between reasoning and in-game events.

**Belief State Representation:**

> Beliefs and reasoning styles are represented as natural-language strategy traces, reflective summaries, and individual reasoning profiles; the framework tests whether models can map behavior to these NL reasoning-style representations.

**Planning Approach:**

> No agent planning module; InMind evaluates whether LLMs can attribute and simulate evolving round-by-round human reasoning, not how an agent itself plans.

**Theory Of Mind Evaluation:**

> Yes — InMind is centrally a theory-of-mind / individualized-reasoning evaluation: tasks like Player Identification, Reflection Alignment, Trace Attribution, and Role Inference jointly measure whether models infer strategic intent and reasoning styles of specific human players.


### Game Loop Design

**Turn Structure:**

> Underlying Avalon turn structure (leader proposal, discussion, voting, quests) is preserved in the recorded sessions; InMind organizes data at the player-turn and round level (884 player turns, 160 strategy traces).

**Phase Handling:**

> Recorded sessions follow standard Avalon phases (team proposal, discussion, approval voting, quest execution); the dataset adds round-level strategy traces and post-game reflections layered on these phases.

**Communication Protocol:**

> Public natural-language discussion plus votes and quest outcomes, captured as transcripts; InMind augments transcripts with private strategy traces and reflective summaries.

**Communication Synchrony:**

> Synchronous turn-based human gameplay as originally played; InMind records the resulting transcripts rather than imposing a new synchrony model.


### Evaluation

**Metrics:**

> Task-specific accuracy/alignment metrics across the four tasks (Player Identification, Reflection Alignment, Trace Attribution, Role Inference), measuring both static alignment and dynamic adaptation of reasoning.

**Persuasion Modeling:**

> Persuasion is not explicitly modeled or optimized; the focus is on capturing and applying individual reasoning styles rather than influence over others.

**Competition Format:**

> Not a competitive arena — InMind is a static benchmark evaluating 11 LLMs on fixed tasks over recorded human games, not LLM-vs-LLM tournaments.

**Llms Evaluated:**

> 11 state-of-the-art LLMs: general-purpose models (Qwen, Yi, GLM-4, InternLM2.5, GPT-4o) and reasoning-enhanced models (DeepSeek-R1, QwQ, o3-mini), among others.

**Human Baseline Comparison:**

> The benchmark is built entirely on annotated human gameplay (the ground truth is human reasoning styles); models are evaluated against human-authored strategy traces, reflections, and identities rather than against live human task performance.

**Data Source Provenance:**

> Curated real human gameplay: 30 full human Avalon sessions (25 participant-mode, 5 observer-mode), 884 player turns, 160 strategy traces, 30 reflective summaries, over 10 hours of play, collected in Mandarin Chinese with participant consent.


### Training Methodology

**Training Paradigm:**

> Tuning-free — InMind evaluates pretrained LLMs via prompting on the four tasks; no SFT or RL is performed.

**Agent Controllability:**

> Not applicable — InMind is an evaluation benchmark, not a trainable agent; no win-rate/difficulty tuning.

**Strategy Space Analysis:**

> Yes — InMind explicitly analyzes diverse but contextually valid individual reasoning strategies, showing different players adopt distinct styles under identical conditions and testing whether models capture this variation.


### Optimization

- **Prompt Optimizer Used:** None — no DSPy/automated prompt optimizer; standardized task prompts are used.
- **Rl Algorithm:** None — no RL or preference optimization.
- **Reward Design:** No training reward; task-level correctness/alignment scores serve only as evaluation signals.
- **Optimization Target:** Not applicable — nothing is optimized; InMind measures existing model capabilities.
- **Self Play Loop:** None — no self-play; the benchmark uses fixed recorded human games.
- **Optimization Cost:** Not applicable — evaluation-only; cost is limited to inference over the 11 evaluated models.

### Engineering

- **Framework Used:** Custom InMind evaluation framework and the InMind-Avalon dataset; not built on DSPy or LangChain.
- **Open Source:** Yes — dataset and evaluation code released at https://github.com/leroy9472/InMind.

**Reproducibility:**

> Good — a fixed annotated dataset with defined tasks and 11 enumerated models supports reproducible evaluation; main constraint is API access to the proprietary models.


### Findings

**Key Results:**

> Evaluating 11 LLMs on InMind-Avalon reveals critical limitations: most models including GPT-4o rely on superficial lexical patterns and fail to consistently infer deeper strategic intent; temporal alignment between reflective reasoning and specific in-game events is hard for nearly all models; and dynamic adaptation of strategic reasoning to evolving interactions is largely insufficient. DeepSeek-R1 performs notably better, showing early signs of style-sensitive reasoning.

**Identified Limitations:**

> Data is collected only in Mandarin Chinese and limited to 30 Avalon sessions; the benchmark covers one game; evaluated models still struggle with temporal alignment and dynamic adaptation, indicating headroom rather than a solved task.


### Uncertain Fields

- Action Tool Design
- Credit Assignment
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Rating System

---

## 25. Memory & agent-architecture references — Generative Agents, ReAct / Reflexion, AgentArch, MemoryAgentBench

### Basic Info

**Name:**

> Memory & agent-architecture references — Generative Agents, ReAct / Reflexion, AgentArch, MemoryAgentBench

**Year:**

> ReAct 2022; Reflexion 2023; Generative Agents 2023 (UIST'23); AgentArch 2025 (arXiv Sep 2025); MemoryAgentBench 2025 (arXiv Jul 2025, ICLR 2026)

**Authors Org:**

> ReAct: Shunyu Yao et al. (Princeton / Google Brain). Reflexion: Noah Shinn, Federico Cassano et al. (Northeastern / MIT / Princeton). Generative Agents: Joon Sung Park, Joseph O'Brien, Carrie Cai, Meredith Ringel Morris, Percy Liang, Michael Bernstein (Stanford / Google DeepMind). AgentArch: Tara Bogavelli, Hari Subramani, Roshnee Sharma (ServiceNow). MemoryAgentBench: HUST-AI group (Yuanzhe Hu et al.; HUST / UCSD collaboration).

**Paper Url:**

> ReAct: https://arxiv.org/abs/2210.03629 ; Reflexion: https://arxiv.org/abs/2303.11366 ; Generative Agents: https://arxiv.org/abs/2304.03442 ; AgentArch: https://arxiv.org/abs/2509.10769 ; MemoryAgentBench: https://arxiv.org/abs/2507.05257

**Repo Url:**

> Generative Agents: https://github.com/joonspk-research/generative_agents ; Reflexion: https://github.com/noahshinn/reflexion ; AgentArch: https://github.com/ServiceNow/AgentArch ; MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench


### Game Coverage

**Games Supported:**

> None of these are social-deduction game systems. Generative Agents simulates a Sims-like sandbox town (Smallville). ReAct evaluates on HotpotQA, FEVER, ALFWorld, WebShop. Reflexion uses ALFWorld, HotpotQA, HumanEval. AgentArch uses enterprise workflows. MemoryAgentBench uses long-context QA/memory tasks. They are reference architectures applicable to building social-game agents, not games themselves.


### Agent Architecture

**Reasoning Paradigm:**

> ReAct introduced interleaved Reasoning + Acting (Thought/Action/Observation traces). Reflexion adds verbal self-reflection: the agent reflects on failed trajectories and stores the reflection to improve subsequent attempts. Generative Agents use LLM-driven reasoning over a retrieved memory stream plus periodic reflection and planning. AgentArch directly compares ReAct prompting against function-calling as agent styles. MemoryAgentBench is paradigm-agnostic — it evaluates memory mechanisms across context-based, RAG, and external-memory agents.

**Memory Mechanism:**

> Generative Agents: the central contribution — a 'memory stream', a natural-language database of timestamped observations; retrieval scores entries by recency, importance, and relevance to the current situation, and periodic 'reflection' synthesizes higher-level memories that are also stored. ReAct: working memory is the running Thought/Action/Observation context window; no long-term store. Reflexion: an episodic memory of natural-language self-reflections appended across trials. AgentArch: explicitly contrasts 'complete' memory (all prior tool calls/responses visible — max context, longer prompts) vs 'summarized' memory (only condensed summaries — shorter context, weaker coordination). MemoryAgentBench: evaluates the spectrum of memory designs — context-based, RAG/retrieval-augmented, and agents with explicit external memory modules and tool integration — via incremental multi-turn information intake.

**Planning Approach:**

> Generative Agents perform explicit hierarchical planning: agents generate a broad daily plan, then recursively decompose it into hour- and minute-level actions, and re-plan reactively when the environment changes. ReAct plans implicitly through interleaved Thought steps. Reflexion plans by revising strategy based on stored reflections between trials. AgentArch evaluates orchestration strategies (single-agent vs orchestrator-led multi-agent) as a planning/coordination dimension. MemoryAgentBench does not focus on planning.

**Action Tool Design:**

> ReAct: actions are environment-specific API calls (e.g., search[], lookup[] for Wikipedia; navigation/manipulation actions for ALFWorld/WebShop) emitted as text. Reflexion: same action spaces as its underlying environments. Generative Agents: actions are natural-language behaviors translated into sandbox movements/interactions. AgentArch: actions are exposed as tools (8 tools in the simple task, 31 in the complex task) and the benchmark specifically compares function-calling tool invocation vs ReAct-style textual tool calls. MemoryAgentBench: memory operations (read/write/retrieve) are the relevant 'actions'.


### Game Loop Design

**Turn Structure:**

> ReAct/Reflexion: per-step Thought->Action->Observation cycles until task completion. Generative Agents: simulation advances on a discrete time clock; each agent acts per tick based on its plan. AgentArch: orchestrator-mediated turn passing among agents. MemoryAgentBench: incremental multi-turn interaction where information chunks arrive turn by turn.

**Communication Protocol:**

> Generative Agents communicate via natural-language dialogue between agents within the sandbox (and to a controlling user). ReAct/Reflexion are single-agent — communication is with the environment/tools only. AgentArch defines inter-agent communication via the orchestrator (isolated agents vs open agent networks) and varies what information is shared (complete vs summarized memory). MemoryAgentBench: user-agent dialogue across multi-turn sessions.


### Evaluation

**Metrics:**

> ReAct: task success / answer accuracy (HotpotQA EM, FEVER accuracy, ALFWorld/WebShop success rate). Reflexion: success rate / pass@1 (e.g., HumanEval 91%, ALFWorld and HotpotQA gains over baselines). Generative Agents: human believability ratings of agent behavior in a controlled evaluation (TrustScore-style human eval). AgentArch: task accuracy (peak 70.8% simple, 35.3% complex) and final-decision quality. MemoryAgentBench: per-competency scores for accurate retrieval, test-time learning, long-range understanding, conflict resolution/selective forgetting.

**Persuasion Modeling:**

> Not modeled. Generative Agents show emergent social influence (e.g., spreading a party invitation) but do not quantify persuasion. ReAct/Reflexion/AgentArch/MemoryAgentBench do not address it.

**Competition Format:**

> Not competitive — no model-vs-model tournament. AgentArch compares architectural configurations head-to-head per model; MemoryAgentBench compares memory-agent systems; comparisons are on shared benchmarks, not adversarial play.

**Data Source Provenance:**

> Generative Agents: synthetic LLM-generated agent behavior in a sandbox, plus human evaluation. ReAct/Reflexion: standard public task datasets/environments. AgentArch: deterministic mock enterprise data with human-annotated ground truth. MemoryAgentBench: repurposed long-context datasets plus two new datasets (EventQA, FactConsolidation).


### Training Methodology

**Training Paradigm:**

> All are tuning-free / prompting-based — no model weight training. ReAct, Reflexion, Generative Agents, AgentArch and MemoryAgentBench all use frozen pretrained LLMs; improvement comes from prompting, memory, reflection, or architecture rather than SFT/RL.


### Optimization

**Rl Algorithm:**

> None — no RL/preference-optimization is used. Reflexion is explicitly framed as an alternative to RL: it improves via verbal self-reflection (linguistic feedback) instead of gradient-based policy updates.

**Reward Design:**

> ReAct: sparse task success signal from the environment. Reflexion: a binary/scalar success signal converted into a natural-language self-reflection that serves as the improvement signal. Generative Agents: an LLM-assigned 'importance' score per memory plus recency/relevance drives retrieval (not a training reward). AgentArch/MemoryAgentBench: task-accuracy/competency scores used for evaluation, not training rewards.

**Optimization Target:**

> ReAct: nothing is optimized (inference-time reasoning). Reflexion: the episodic reflection memory is iteratively improved across trials. Generative Agents: the memory stream and plans are continually updated, not optimized via search. AgentArch: compares fixed architectural designs. MemoryAgentBench: compares fixed memory mechanisms. No prompt/weight optimization target.


### Engineering

**Framework Used:**

> Custom research code per project. Generative Agents: a custom Python sandbox with a memory-stream architecture. ReAct/Reflexion: custom prompting harnesses over LLM APIs. AgentArch: a custom benchmark harness (ServiceNow). MemoryAgentBench: a custom unified evaluation suite. None use DSPy/LangChain as the core, though concepts are widely reimplemented in LangChain/DSPy.

**Open Source:**

> Yes for all — Generative Agents, Reflexion, AgentArch and MemoryAgentBench release code on GitHub; ReAct released prompts/code. Datasets: AgentArch provides mock data + prompts; MemoryAgentBench releases EventQA and FactConsolidation.

**Reproducibility:**

> Moderate to good. Generative Agents: code public but costly and LLM-version-sensitive to reproduce. ReAct/Reflexion: prompts and code public, widely re-implemented. AgentArch: deterministic mock data, temperature 0, human-annotated ground truth and appendix prompts aid reproducibility. MemoryAgentBench: standardized protocol and released datasets support reproduction; LLM API drift remains a risk.


### Findings

**Key Results:**

> Generative Agents introduced the memory-stream + reflection + planning architecture and showed believable emergent social behavior (e.g., agents autonomously organizing a Valentine's party, spreading information); human raters found agent behavior believable, and ablating memory/reflection/planning degraded believability. ReAct showed interleaving reasoning and acting beats reasoning-only or acting-only on QA and decision tasks and reduces hallucination. Reflexion showed verbal self-reflection yields large gains (e.g., 91% pass@1 on HumanEval) without weight updates. AgentArch found no single best architecture — optimal configuration is model- and task-dependent, peak accuracy was only 70.8% (simple) / 35.3% (complex), function-calling generally beat ReAct, and multi-agent ReAct consistently underperformed, while multi-agent setups produced better final decisions despite lower overall accuracy. MemoryAgentBench found that no current memory agent masters all four competencies (accurate retrieval, test-time learning, long-range understanding, selective forgetting/conflict resolution), motivating better memory mechanisms.

**Identified Limitations:**

> Generative Agents: high LLM cost, retrieval can miss relevant memories, occasional embellished/hallucinated memories, sensitivity to prompt and model version. ReAct: dependent on a strong base model and well-designed action space, can loop or get stuck. Reflexion: requires informative success signals and a capable reflection model; limited by context length for accumulated reflections. AgentArch: only two enterprise use cases, limited model diversity (one open-source, one reasoning model), text-only, temperature fixed at 0, no latency/cost metrics. MemoryAgentBench: current memory methods fall short on all four competencies; benchmark partly built from repurposed datasets.


### Uncertain Fields

- Agent Controllability
- Belief State Representation
- Communication Synchrony
- Credit Assignment
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Hidden Role Mechanic
- Human Baseline Comparison
- Llms Evaluated
- Multimodal Support
- Num Players
- Optimization Cost
- Phase Handling
- Prompt Optimizer Used
- Rating System
- Rule Variant Coverage
- Self Play Loop
- Strategy Space Analysis
- Theory Of Mind Evaluation

---

## 26. Multi-agent game-environment libraries — OpenSpiel, PettingZoo, ChatArena, TextArena

### Basic Info

- **Name:** Multi-agent game-environment libraries — OpenSpiel, PettingZoo, ChatArena, TextArena
- **Year:** OpenSpiel 2019; PettingZoo 2020; ChatArena 2023 (deprecated Aug 2025); TextArena 2025

**Authors Org:**

> OpenSpiel: Google DeepMind (Lanctot et al.). PettingZoo: Farama Foundation (J.K. Terry et al.). ChatArena: Farama Foundation (Yuxiang Wu, Zhengyao Jiang, Akbir Khan, Yao Fu, Laura Ruis, Edward Grefenstette, Tim Rocktäschel). TextArena: A*STAR / collaborators (Leon Guertler, Bobby Cheng, Simon Yu, Bo Liu, Leshem Choshen, Cheston Tan).

**Paper Url:**

> OpenSpiel: https://arxiv.org/abs/1908.09453 ; PettingZoo: https://arxiv.org/abs/2009.14471 ; TextArena: https://arxiv.org/abs/2504.11442 ; ChatArena: no formal paper (project README/seminar)

**Repo Url:**

> OpenSpiel: https://github.com/google-deepmind/open_spiel ; PettingZoo: https://github.com/Farama-Foundation/PettingZoo ; ChatArena: https://github.com/Farama-Foundation/chatarena ; TextArena: https://github.com/LeonGuertler/TextArena (also github.com/TextArena/TextArena)


### Game Coverage

**Games Supported:**

> OpenSpiel: 70+ classical/board/card games incl. imperfect-information games (Poker, Hanabi, Hearts), grid-world social dilemmas; no dedicated Werewolf/Mafia env in core. PettingZoo: Atari, Butterfly, Classic (chess, card games); no social-deduction games. ChatArena: Chameleon (hidden-role deduction), NLP Classroom, RPS, Tic-tac-toe, PettingZoo chess. TextArena: 57+ (later 74+) text games incl. SecretMafia, Two Rooms and a Boom, Diplomacy, TruthAndDeception, Negotiation, KuhnPoker, Debate.

**Num Players:**

> OpenSpiel: n-player (single- and multi-agent). PettingZoo: 2 to many (Atari/Butterfly). ChatArena: 2+ players. TextArena: 1-15 players (16 single-, 47 two-, 11 multi-player envs).

**Hidden Role Mechanic:**

> Library-level: OpenSpiel models imperfect information via information-state abstractions and chance nodes in extensive-form games; PettingZoo exposes per-agent partial observations via the AEC API. ChatArena's Chameleon implements a single hidden-role player who must blend in. TextArena's SecretMafia gives mafia members shared private knowledge while villagers see only public chat; hidden roles are conveyed via private text observations.

**Rule Variant Coverage:**

> OpenSpiel: very broad (70+ games, many parameterizable). PettingZoo: ~60+ environments across 3 families. ChatArena: ~5 environments. TextArena: 57+/74+ environments with configurable player counts and variants; breadth is the explicit selling point.


### Agent Architecture

**Reasoning Paradigm:**

> Environment libraries, not agents — they are reasoning-agnostic. OpenSpiel/PettingZoo target RL policies, search and planning (MCTS, CFR). ChatArena and TextArena are designed to host LLM agents using prompting/ReAct-style reasoning; the agent reasoning approach is left to the user.

**Memory Mechanism:**

> Provided as environment-side state/observation history rather than agent memory. OpenSpiel exposes information-state strings/tensors that encode the action history visible to a player. PettingZoo's AEC API yields per-step observations; history tracking is the user's responsibility. ChatArena and TextArena pass accumulated text transcripts (public chat logs) as observations; neither ships a built-in agent memory store — memory architecture (summary, external store, retrieval) must be added by the user.

**Planning Approach:**

> Library-dependent. OpenSpiel ships search/planning algorithms (MCTS, AlphaZero, CFR variants). PettingZoo provides only the environment API; planning is up to the chosen RL/search algorithm. ChatArena and TextArena provide no planner — agents plan implicitly through LLM prompting.

**Action Tool Design:**

> OpenSpiel: discrete legal-action lists per state via a procedural extensive-form interface. PettingZoo: Gym-style action spaces per agent under the AEC cycle (agent_iter/step). ChatArena: agents emit free-form natural-language messages parsed by the environment moderator. TextArena: Gym-compatible step()/observation API; actions are submitted as text strings parsed by each environment. Actions are not exposed as named callable tools/functions — that mapping is left to the agent framework (e.g., DSPy ReAct tools).

**Theory Of Mind Evaluation:**

> Not measured directly by the libraries. TextArena profiles soft skills including a 'Theory of Mind' category across its social/bluffing games and reports per-skill model scores, providing an indirect ToM signal. OpenSpiel/PettingZoo/ChatArena do not include explicit ToM metrics.


### Game Loop Design

**Turn Structure:**

> OpenSpiel: strictly sequential or simultaneous-move depending on the game; turn order defined by the extensive-form game tree. PettingZoo: AEC model cycles agents one at a time (agent_iter) plus a Parallel API for simultaneous steps. ChatArena: turn-based or parallel, optionally mediated by a Moderator. TextArena: environment-managed turn sequencing; one player acts per step in most envs.

**Phase Handling:**

> Phase logic is encoded inside each individual game/environment, not the library core. TextArena's SecretMafia and similar multi-player envs internally manage discussion/voting/night phases. ChatArena's Moderator component can enforce phase transitions. OpenSpiel encodes phases as nodes in the game tree; PettingZoo encodes them in per-env step logic.

**Communication Protocol:**

> OpenSpiel/PettingZoo: communication only as in-game actions/messages where a game defines them (e.g., Hanabi hints); no general chat channel. ChatArena: public natural-language messages, optionally moderated, with support for private/role channels. TextArena: text observations and text actions; public chat plus environment-delivered private information (e.g., mafia knowledge); voting handled as structured text actions.


### Evaluation

**Metrics:**

> OpenSpiel: game-theoretic metrics (exploitability/NashConv, returns). PettingZoo: cumulative rewards per agent. ChatArena: no standardized metric — qualitative interaction logs. TextArena: TrueSkill rating per model plus per-soft-skill profiles (strategic planning, theory of mind, logical reasoning, etc.) and win/loss outcomes.

**Persuasion Modeling:**

> Not explicitly modeled or optimized by any of the four libraries. Persuasion can be observed emergently in TextArena's negotiation/social games and ChatArena's Chameleon but is not measured as a separate quantity.

**Rating System:**

> OpenSpiel: exploitability/NashConv (not a competitive ladder). PettingZoo: none built in. ChatArena: none. TextArena: TrueSkill (Bayesian, μ=25, σ=25/3) with an online real-time leaderboard, chosen over Elo for variable player counts and faster convergence.

**Competition Format:**

> OpenSpiel/PettingZoo: support self-play and cross-play training; no managed tournament. ChatArena: ad-hoc multi-agent matches. TextArena: model-vs-model and model-vs-human play with an online leaderboard; 283 models ranked (64 officially hosted plus community submissions); a collective 'Humanity' baseline.

**Llms Evaluated:**

> OpenSpiel/PettingZoo: not LLM-focused (RL agents). ChatArena: backend-agnostic, supports OpenAI/Anthropic/Cohere/HuggingFace models; no systematic study. TextArena: frontier LLMs incl. GPT-4o, Claude-3.5-Sonnet, Gemini, DeepSeek-R1 and 280+ models via OpenRouter.

**Human Baseline Comparison:**

> OpenSpiel and PettingZoo: no human baselines. ChatArena: none. TextArena: yes — supports model-vs-human online play and aggregates human performance into a 'Humanity' TrueSkill baseline compared against frontier models.

**Data Source Provenance:**

> OpenSpiel/PettingZoo: synthetic agent-vs-agent / RL self-play trajectories. ChatArena: synthetic LLM-vs-LLM interaction logs. TextArena: both synthetic LLM-vs-LLM matches and LLM-vs-human online gameplay.


### Training Methodology

**Training Paradigm:**

> OpenSpiel and PettingZoo are training-platform agnostic and used for RL/search (CFR, PPO, AlphaZero). ChatArena targets tuning-free prompting of LLM agents. TextArena is positioned for both evaluation of tuning-free LLMs and as an RL environment suite for training language models via competitive gameplay.


### Optimization

**Prompt Optimizer Used:**

> None — these are environment libraries, not optimizers. They are compatible with external optimizers (e.g., DSPy GEPA/MIPROv2) applied to the hosted agents, but no optimizer ships with them.

**Rl Algorithm:**

> OpenSpiel ships many RL/equilibrium algorithms (CFR, Deep CFR, NFSP, PSRO, PPO, AlphaZero, MCTS). PettingZoo integrates with external MARL libraries (RLlib, Tianshou, Stable-Baselines3). ChatArena: none. TextArena: no built-in RL algorithm but explicitly designed as an RL environment suite for training LMs.

**Reward Design:**

> OpenSpiel/PettingZoo: per-game numeric returns (win/loss, zero-sum or general-sum payoffs) defined by each environment. ChatArena: no standardized reward. TextArena: game-outcome (win/loss) signals per environment, usable as RL reward; no dense auxiliary listening/speaking rewards built in.

**Optimization Target:**

> Not applicable at the library level — what is optimized depends on the user's algorithm (policy weights for RL on OpenSpiel/PettingZoo; prompts/instructions for LLM agents hosted in ChatArena/TextArena).

**Self Play Loop:**

> OpenSpiel and PettingZoo natively support self-play and population-based training. TextArena supports self-play and cross-play matchmaking for both evaluation and RL data generation. ChatArena has no managed self-play loop.


### Engineering

**Framework Used:**

> OpenSpiel: C++ core with Python bindings. PettingZoo: Python, Gymnasium-style AEC API. ChatArena: Python, MDP-based modular framework (Arena/Environment/Backend/Player). TextArena: Python, OpenAI Gym/Gymnasium-inspired API with stackable wrappers.

**Reproducibility:**

> High for OpenSpiel and PettingZoo — mature, well-documented, widely used, standardized APIs and reference environments. ChatArena: moderate, with docs and examples but now deprecated and unmaintained. TextArena: good — documentation at textarena.ai, GitHub, community-extensible envs and an online leaderboard.


### Findings

**Key Results:**

> OpenSpiel and PettingZoo became de facto standards for game-based RL / MARL research, providing standardized APIs (extensive-form games; the AEC model) that decouple environments from algorithms. ChatArena demonstrated LLM multi-agent language games (incl. the hidden-role game Chameleon) but was deprecated in Aug 2025 for lack of adoption. TextArena evaluated 283 models across 57+ text games with TrueSkill, profiled 10 soft skills, and found that frontier reasoning models can leak their own hidden roles during play, degrading social-deduction performance.

**Identified Limitations:**

> OpenSpiel/PettingZoo: not designed for LLM/language-based agents and lack native social-deduction environments. ChatArena: small environment set, no standardized metrics, deprecated/unmaintained. TextArena: classic titles like Werewolf/Secret Hitler not explicitly implemented (SecretMafia is the closest); some environments incomplete; results conflate rule-understanding with strategic skill; synchronous-only communication.


### Uncertain Fields

- Agent Controllability
- Belief State Representation
- Communication Synchrony
- Credit Assignment
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Multimodal Support
- Open Source
- Optimization Cost
- Strategy Space Analysis

---

## 27. TextGrad — Automatic "Differentiation" via Text

### Basic Info

- **Name:** TextGrad — Automatic "Differentiation" via Text
- **Year:** 2024 (arXiv June 2024; published in Nature 2025)

**Authors Org:**

> Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin, James Zou — Stanford University (Computer Science, Biomedical Data Science) and Chan Zuckerberg Biohub.

- **Paper Url:** https://arxiv.org/abs/2406.07496 (also published in Nature, 2025)
- **Repo Url:** https://github.com/zou-group/textgrad

### Agent Architecture

**Reasoning Paradigm:**

> Not a reasoning agent — TextGrad is an optimizer. It uses an LLM-as-a-gradient metaphor: an LLM produces natural-language criticism ('textual gradients') that is backpropagated through a computation graph; optimization analogizes to gradient descent (Textual Gradient Descent, TGD).

**Memory Mechanism:**

> No agent memory. TextGrad maintains a computation graph of Variables; intermediate variables and the textual gradients attached to them are the persisted state during an optimization run. Feedback from every context in which a variable is used is aggregated onto that variable.

**Planning Approach:**

> No planning module. Optimization proceeds iteratively: forward pass, loss evaluation, textual-gradient backward pass, then a TGD update step; typically 3-12 iterations. Variance-reduction techniques (e.g., momentum, multiple gradients) are used to stabilize updates.

**Action Tool Design:**

> Not a tool-using agent. TextGrad exposes a PyTorch-like API — Variable, TextLoss, BlackboxLLM, and a TGD optimizer with .backward() and .step() — so functions/LLM calls/simulators are nodes in a graph rather than agent-callable tools. Authors note extending to tool-use systems is future work.


### Evaluation

**Metrics:**

> Task-specific accuracy/quality metrics used as the loss signal: QA accuracy (GPQA, MMLU), code completion rate (LeetCode Hard), reasoning accuracy (BBH object counting, word sorting), GSM8k math accuracy, plus in-silico binding affinity / druglikeness and radiotherapy dose metrics.

**Llms Evaluated:**

> GPT-4o (primary engine for producing textual gradients/feedback) and GPT-3.5-turbo-0125 (student model optimized via GPT-4o feedback); the library also supports other LLMs as engines.

**Data Source Provenance:**

> Standard public task datasets and benchmarks (GPQA, MMLU subsets, BBH, GSM8k, LeetCode Hard) plus protein-target sets for molecule design and prostate-cancer radiotherapy cases; no game-play data.


### Training Methodology

**Training Paradigm:**

> Tuning-free / inference-time optimization — TextGrad optimizes prompts, instructions or solution instances via LLM textual feedback without updating model weights (no SFT/RL on the model). It performs two modes: instance optimization (one solution) and prompt optimization (generalizing prompt).


### Optimization

**Prompt Optimizer Used:**

> TextGrad IS the optimizer. It is positioned as a comparison point against DSPy and ProTeGi: DSPy focuses on bootstrapped few-shot demonstration search; ProTeGi applies textual gradients to prompts only; TextGrad generalizes backpropagation of natural-language feedback across whole compound systems (prompts, code, molecules, treatment plans). It reports matching or beating DSPy on reasoning tasks while using instruction-only, zero demonstrations.

**Rl Algorithm:**

> None — TextGrad uses no RL/preference-optimization algorithm; Textual Gradient Descent is a metaphorical, LLM-driven analogue of gradient descent rather than PPO/DPO/GRPO.

**Reward Design:**

> The training signal is a TextLoss — a natural-language evaluation of an output (e.g., a critique of an answer, code test results, a binding/druglikeness score, dose conformity). This loss is converted by an LLM into textual gradients; the signal is interpretable natural-language criticism rather than a scalar reward alone.

**Optimization Target:**

> Any text Variable in the computation graph: prompts/system instructions, solution instances (code snippets, answers), molecular structures (SMILES), and radiotherapy treatment-plan parameters. Model weights are NOT a target.

**Credit Assignment:**

> Credit is assigned via textual gradients propagated backward through the computation graph: each variable accumulates feedback from every downstream context in which it was used, enabling per-variable (per-component) credit assignment across multi-step compound systems without numerical differentiation.

**Optimization Cost:**

> Each backward pass costs at most n additional LLM calls for a graph with n edges; runs typically converge in 3-12 iterations. Cost is modest relative to weight training but scales with graph size, batch size and iteration count; variance-reduction may add calls.


### Engineering

**Framework Used:**

> TextGrad — a Python library following PyTorch's syntax and abstractions (Variable, loss, .backward(), optimizer.step()); LLM engines (GPT-4o, GPT-3.5) accessed as black-box components.

**Open Source:**

> Yes — open source at github.com/zou-group/textgrad (MIT license); the method was published in Nature.

**Reproducibility:**

> Good — public PyTorch-style library, documented API and example notebooks for the reported tasks. Authors caution that optimization stability can require variance-reduction techniques and that some applications are in-silico proof-of-concept needing further validation.


### Findings

**Key Results:**

> TextGrad backpropagates natural-language feedback to optimize arbitrary compound AI systems. Headline results: GPQA (Google-Proof QA) accuracy raised for GPT-4o from ~51% to ~55%; ~20% relative gain on LeetCode-Hard code solutions (36% completion vs 31% for Reflexion); MMLU Machine Learning 85.7%->88.4% and College Physics 91.2%->95.1%; competitive in-silico drug molecules across 58 protein targets; clinically plausible prostate radiotherapy plans. It matches/exceeds DSPy on reasoning using instruction-only optimization with zero demonstrations, making it a strong comparison point against GEPA and MIPROv2.

**Identified Limitations:**

> Results for medicine/chemistry are in-silico proof-of-concept needing experimental/clinical validation; optimization can be unstable and require variance-reduction; not yet extended to tool-use and retrieval-augmented systems; performance depends on a capable feedback LLM; gains on some benchmarks are modest (a few accuracy points).


### Uncertain Fields

- Agent Controllability
- Belief State Representation
- Communication Protocol
- Communication Synchrony
- Competition Format
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Games Supported
- Hidden Role Mechanic
- Human Baseline Comparison
- Multimodal Support
- Num Players
- Persuasion Modeling
- Phase Handling
- Rating System
- Rule Variant Coverage
- Self Play Loop
- Strategy Space Analysis
- Theory Of Mind Evaluation
- Turn Structure

---

## 28. The Stackelberg Speaker — Optimizing Persuasive Communication in Social Deduction Games (also titled 'Leading the Follower: Learning Persuasive Agents in Social Deduction Games')

### Basic Info

**Name:**

> The Stackelberg Speaker — Optimizing Persuasive Communication in Social Deduction Games (also titled 'Leading the Follower: Learning Persuasive Agents in Social Deduction Games')

- **Year:** 2025 (arXiv October 2025; revised April 2026; ICLR 2026)

**Authors Org:**

> Zheng Zhang, Deheng Ye, Peilin Zhao, Hao Wang. Affiliations: Hong Kong University of Science and Technology (Guangzhou), Tencent, Shanghai Jiao Tong University.

- **Paper Url:** https://arxiv.org/abs/2510.09087
- **Repo Url:** https://3dagentworld.github.io/leader_follower

### Game Coverage

**Games Supported:**

> Three social deduction games — Werewolf, The Resistance: Avalon, and One Night Ultimate Werewolf (ONUW) — plus Sotopia, a social-simulation environment, for generalization tests.

- **Num Players:** Werewolf: 7 players; Avalon: 5 players; ONUW: 5 players; Sotopia: 2 agents.

**Hidden Role Mechanic:**

> Standard hidden-role asymmetric-information structure of each game (werewolves/evil minions vs villagers/good with special roles); the paper's focus is on persuasive communication given these hidden roles rather than on the role mechanic itself.

**Rule Variant Coverage:**

> Three distinct social deduction games with different role sets (Werewolf, Avalon, ONUW), plus a non-SDG social environment (Sotopia) — moderate cross-game generalization breadth.

- **Multimodal Support:** Text-only; turn-based dialogue. No audio/video/prosody modalities.

### Agent Architecture

**Reasoning Paradigm:**

> Game-theoretic RL framing: turn-based dialogue is modeled as a Stackelberg (leader-follower) competition. A two-stage pipeline uses an API LLM to generate a base utterance and a fine-tuned Refiner (RL policy) to optimize it for persuasive impact.

**Belief State Representation:**

> Persuasive intent is operationalized as desired vs undesired follower responses; influence is represented via shifts in the follower's response distribution (log-probability shifts) rather than an explicit role-probability belief table.

**Planning Approach:**

> Persuasive planning via the leader optimizing utterances to steer the next player's (follower's) response toward a desired outcome; no tree search — strategy is learned through GRPO fine-tuning of the Refiner.

**Action Tool Design:**

> Game actions and dialogue come from the underlying SDG agent frameworks; the Stackelberg Speaker adds a Refiner module that rewrites/optimizes utterances. Persuasion is decomposed into Intent Identification, Impact Measurement, and Strategy Optimization components rather than discrete tool calls.

**Theory Of Mind Evaluation:**

> Partial — the framework reasons about how an utterance changes the follower's response distribution, an implicit model of the other agent's mind, but it does not present a dedicated standalone ToM benchmark.


### Game Loop Design

**Turn Structure:**

> Turn-based dialogue: each turn the current speaker (leader) produces an utterance that influences the next speaker (follower); modeled as a sequential leader-follower game.

**Phase Handling:**

> Discussion phases of each SDG are the focus (where persuasive utterances matter); standard day/night, voting, and quest phases of Werewolf/Avalon/ONUW are handled by the underlying game frameworks.

**Communication Protocol:**

> Public turn-based natural-language discussion plus votes; the leader's utterance is crafted to shift the follower's subsequent public response.

**Communication Synchrony:**

> Synchronous turn-based: the Stackelberg formulation depends on a fixed leader-then-follower turn order rather than asynchronous or bidding-based speech.


### Evaluation

**Metrics:**

> Faction/team win rates per game, human-evaluation win rate and votes received, goal-completion scores (Sotopia), and ablations on reward components and the K hyperparameter.

**Persuasion Modeling:**

> Yes — persuasion is the central, explicitly optimized objective. It is decomposed into Intent Identification (K=3 desired and undesired responses), Impact Measurement (response-distribution change), and Strategy Optimization (GRPO fine-tuning), directly optimizing utterances for persuasive impact.

**Competition Format:**

> Cross-play / head-to-head: Stackelberg-Speaker-augmented agents are pitted against baseline agent frameworks (LSPO, Strategist, RL-instructed) within each game, plus human-evaluation games.

**Llms Evaluated:**

> Backend/evaluation LLMs: GPT-4o, GPT-5, Gemini-2.5-Flash, Claude-3.5-Haiku. Fine-tuned Refiner: Qwen2.5-7B-Instruct (LoRA rank 16). Frozen Measurer: Qwen2.5-72B-Instruct.

**Human Baseline Comparison:**

> Yes — a human evaluation pits the agents against or alongside human players; the Stackelberg Speaker agent achieves the highest win rate (44.1%) and receives the fewest votes against it.

**Data Source Provenance:**

> Synthetic LLM-vs-LLM gameplay across Werewolf/Avalon/ONUW for training and evaluation, supplemented by a human-evaluation study (LLM-vs-human games).


### Training Methodology

**Training Paradigm:**

> Reinforcement learning fine-tuning: a frozen API LLM produces base utterances and a Qwen2.5-7B Refiner is fine-tuned with GRPO (LoRA) to optimize persuasive impact.

**Strategy Space Analysis:**

> Persuasive-communication strategy is analyzed via ablations on reward components and K; the work shows persuasion-optimized agents adopt more influential utterance strategies than baselines.


### Optimization

**Prompt Optimizer Used:**

> None — no DSPy/automated prompt optimizer; the persuasion gain comes from RL fine-tuning of the Refiner, not prompt optimization.

**Rl Algorithm:**

> Group Relative Policy Optimization (GRPO), which computes relative advantages within a group of sampled responses without an explicit critic model; applied with KL regularization.

**Reward Design:**

> Persuasion reward based on log-probability shifts: maximizing the probability of desired follower responses while minimizing undesired ones, scored by a frozen Measurer (Qwen2.5-72B-Instruct) over K=3 desired and K=3 undesired target responses.

**Optimization Target:**

> Policy weights of the Refiner LLM (Qwen2.5-7B-Instruct via LoRA) — i.e., the utterance-refinement policy is optimized, not prompts or demonstrations.

**Credit Assignment:**

> Credit is assigned at the single-turn utterance level via the immediate persuasion reward (follower response-distribution shift measured by the frozen Measurer), converting delayed game outcomes into per-utterance signals; GRPO then uses group-relative advantages.


### Engineering

**Framework Used:**

> ms-swift (ModelScope) for GRPO fine-tuning; the agents are layered on existing SDG agent frameworks (LSPO, Strategist, RL-instructed) plus the Sotopia environment. Not DSPy-based.

- **Open Source:** Code/resources available via the project page (https://3dagentworld.github.io/leader_follower).

**Reproducibility:**

> Moderate — the paper specifies the GRPO setup, ms-swift framework, LoRA rank, base/Measurer models, and reward design; full reproducibility depends on released code and access to the API backend LLMs.


### Findings

**Key Results:**

> Adding the Stackelberg Speaker improves win rates across all three games: Werewolf 44.7% (vs 38.6% LSPO baseline), Avalon 61.3% (vs 57.4% Strategist), ONUW 51.5% (vs 48.5% RL-instructed). In human evaluation the persuasion-optimized agent attains the highest win rate (44.1%) and receives the fewest votes against it. Ablations confirm each reward/persuasion component contributes.

**Identified Limitations:**

> Training tends to increase the length of generated utterances, indicating a need for future length-penalty mechanisms; gains are incremental and depend on the underlying agent frameworks and a strong frozen Measurer.


### Uncertain Fields

- Agent Controllability
- Deceiver Detector Asymmetry
- Deception Emergence
- Deception Metric Design
- Memory Mechanism
- Optimization Cost
- Rating System
- Self Play Loop

---
