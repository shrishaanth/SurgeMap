# Presentation: Member 2's part

Slides 10 to 15 of the 15-minute talk. Member 1 has covered the problem, the classical methods,
our approach and the whole spatial part (the graph, the graph convolution and the learned graph).
Member 2 explains the **temporal part**, shows **how all the components connect**, presents the
**final system architecture**, and closes with **what we expect to do next**. This part has no
results section by design.

**Time budget: about 6.5 minutes.** The scripts total roughly 815 words, which is about 6.3
minutes at a spoken pace of 130 words a minute. Replace `[Name 1]` and `[Name 2]` with your names.

| # | Slide | Minutes |
|---|-------|---------|
| 10 | The temporal part: why it is needed and what it captures | 1.0 |
| 11 | How the network reads time: dilated convolutions | 1.3 |
| 12 | Deciding what passes through: gates, residual and skip connections | 1.0 |
| 13 | How all the components connect | 1.2 |
| 14 | The final system architecture | 1.0 |
| 15 | What we expect to do next | 0.8 |
| | **Total** | **6.3** |

---

## Slide 10: The temporal part: why it is needed and what it captures

**Title:** Reading each zone's recent history

**On the slide**
- **Why:** the next hour of demand in a zone depends on what that zone has been doing
- **What the network is given, per zone and per 5-minute step** (7 numbers, 48 steps = 4 hours)

| Input | Why |
|---|---|
| Pickups, log-scaled and standardised per zone | counts are skewed; a quiet and a busy zone end up on the same scale |
| Hour of day, as a sine and a cosine | so that 23:55 and 00:00 are neighbours |
| Day of week, as a sine and a cosine | the same, for Sunday and Monday |
| Weekend flag | weekends have their own rhythm |
| Dropoffs, log-scaled and standardised | passengers arriving now are tomorrow's pickups nearby |

- **Patterns it is there to capture**
  - short-term momentum: climbing for the last 20 minutes
  - the daily cycle: peaks and quiet hours
  - the weekly cycle: a Friday night differs from a Tuesday night
  - departures from routine: twice the usual level for this hour
- Plus, at the output: demand at the target time **one day and one week earlier**

**Visual:** a 4-hour window of one zone's pickups with the next hour as a shaded forecast region.

**Script**

> "The temporal part answers a simple question: what has this zone been doing? For every zone and
> every five-minute step, the network sees seven numbers: the pickups, scaled so busy and quiet
> zones are comparable; the time of day and day of week, written as sines and cosines so that
> midnight sits next to eleven fifty-five; a weekend flag; and the dropoffs, because passengers
> arriving now are pickups nearby soon. We give it forty-eight steps, four hours.
>
> That is enough to see momentum, the daily cycle, and when a zone is running away from its
> routine. Four hours cannot show a weekly rhythm, so at the end we also give the network the demand
> at the target time one day earlier and one week earlier."

---

## Slide 11: How the network reads time: dilated convolutions

**Title:** Looking minutes and hours back with a few small filters

**On the slide**
- **A temporal convolution** slides a small learned filter along time. Ours looks at **two** steps:
  now, and *d* steps back.
- **Dilation *d*** sets the gap. Across the 8 layers: **1, 2, 4, 8, 16, 1, 2, 4**
- **They stack:** layer 1 combines neighbours, layer 2 combines pairs of those, and so on, so each
  output reaches further back than any single filter
- **Reach:** 1 + (1 + 2 + 4 + 8 + 16 + 1 + 2 + 4) = **39 steps = 3 hours 15 minutes**, inside the
  48-step window
- **Causal:** every output uses only the present and earlier steps, never later ones
- **The time axis shrinks** at each layer by its dilation: 48 → 47 → 45 → 41 → 33 → 17 → 16 → 14 →
  10. The newest step holds the summary.
- **Versus a GRU** (our first design): a GRU walks through the steps one by one and carries a
  memory; here every step is read directly, at several time scales

**Visual:** the dilation tree: the output at the top, arrows back 1, 2, 4, 8, 16 steps through the
layers.

**Script**

> "A convolution over time is a small filter that slides along the sequence. Ours looks at two
> steps: the current one, and one some distance back. That distance is the dilation. The eight
> layers use dilations of one, two, four, eight, sixteen, then one, two, four again.
>
> The point is that they stack. The first layer combines adjacent steps; the second combines pairs
> of those, two apart; and so on. Add up the gaps and each final output reaches back thirty-nine
> steps, about three hours fifteen, using only eight small layers. Early layers see the last few
> minutes; later ones see slow change over hours.
>
> It is causal: an output never uses a later step. And the time axis shrinks as we go, from
> forty-eight steps to ten, which is why the newest step is the one that carries the summary.
>
> Our first design used a GRU, which walks through the steps one at a time and has to remember.
> Here every step is read directly."

---

## Slide 12: Deciding what passes through: gates, residual and skip connections

**Title:** Gates, shortcuts and summaries

**On the slide**
- **Gate** (inside each temporal layer)
  - two filters read the same input: one proposes values, the other, squashed to between 0 and 1,
    decides **how much of each to let through**
  - output = proposal x gate
  - lets the network ignore history that does not matter right now
- **Residual connection**
  - the layer's input is added back onto its output
  - the layer learns a **correction**, not a rewrite; and signals and learning can flow through a
    deep stack
- **Normalisation** keeps the numbers well scaled between layers
- **Skip connection**
  - each of the 8 layers sends a 128-number summary of its **newest time step** straight to the
    output
  - the output therefore sees fine detail from early layers and slow patterns from late ones

**Visual:** one layer as a block diagram: input → (filter, gate) → multiply → graph mixing → add
input → normalise → output; with a side arrow from the output to a "skip sum" box.

**Script**

> "Three ideas control how information moves. First, the gate. Inside each temporal layer, two
> filters read the same input. One proposes values; the other, squashed between zero and one,
> decides how much of each to let through. Multiply them. That lets the network switch off history
> that is irrelevant at this moment.
>
> Second, the residual connection: the layer's input is added back onto its output. So each layer
> only has to learn a correction, and signals and learning can travel through a deep stack.
>
> Third, skip connections. Each of the eight layers sends a short summary of its newest time step
> straight to the output. The output therefore sees what the early layers noticed, fine detail, and
> what the late layers noticed, slow patterns, all at once."

---

## Slide 13: How all the components connect

**Title:** One forward pass, from history to forecast

**On the slide**

| Step | What happens | Shape (zones x steps x channels) |
|---|---|---|
| Input | 7 numbers per zone per step | 258 x 48 x 7 |
| Projection | a 1x1 layer lifts them to 32 channels | 258 x 48 x 32 |
| **Layer, repeated 8 times** | gated temporal convolution, then graph convolution *(Member 1)*, then add the input back and normalise; send a 128-number summary of the newest step to the skip sum | time axis shrinks 48 → 10 |
| Skip sum | add the 8 summaries | 258 x 128 |
| Output layer | one 128-to-128 layer with ReLU activations | 258 x 128 |
| Four heads | one per horizon, each also given demand at the target time one day and one week earlier | 258 x 4 |
| Convert | standardised log-demand back to a pickup count | 258 x 4 |

- **Time and space alternate**, 8 times: a zone passes on what it has worked out, not only raw counts
- **One loss, one network:** trained end to end with a Poisson loss on the pickup counts, so the
  filters, gates, graph mixing and learned graph all learn together
- **147,644 parameters**

| Where the parameters are | Count | Share |
|---|---|---|
| Graph convolution layers | 57,600 | 39% |
| Skips and output layers | 50,304 | 34% |
| Temporal filters and gates | 33,280 | 23% |
| Heads, projection, normalisation, learned graph | 6,460 | 4% |

**Visual:** the forward-pass flowchart with the shapes written on the arrows.

**Script**

> "Now let's put the pieces together. Input is two hundred fifty-eight zones, forty-eight steps,
> seven numbers. A projection lifts that to thirty-two channels. Then one layer, repeated eight
> times: a gated temporal convolution reads the history, then my partner's graph convolution lets
> zones exchange what they have found, then the input is added back and normalised, and a summary of
> the newest step is sent to the side.
>
> After eight layers, those summaries are added up, passed through an output layer, and handed to
> four heads, one per horizon. Each head also receives the demand at the target time one day and one
> week earlier. What comes out is a standardised log-demand per zone, which we convert back into a
> pickup count.
>
> The whole thing is trained at once with one loss, a Poisson loss on the counts. The graph mixing
> is the largest piece, 39% of the 147,644 parameters; temporal filters and gates are 23%."

---

## Slide 14: The final system architecture

**Title:** From raw trips to forecasts and decisions

**On the slide**

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Download | `download` | NYC TLC website | monthly trip files |
| Preprocess | `preprocess` | trip files | 5-minute demand per zone, features, flow graph, travel times, split dates |
| Train | `train` | processed data | three Graph WaveNet checkpoints |
| Forecast | `export`, `baselines` | checkpoints, data | every model's forecasts, in one file |
| Calibrate | `calibrate`, `align` | forecasts | corrected forecasts on the test days |
| Analyse | `headroom`, `regimes` | forecasts | error analyses |
| Simulate | `simulate`, `plots` | forecasts, travel times | fleet study and figures |

- **One Python package** (`surgemap`), one command per stage; stages communicate **only through files**
- **Layers:** data, models, training, evaluation, simulation
- **Design decisions**
  - **training is separated from everything else:** a GPU is needed once; every later stage runs on
    a laptop from saved forecasts
  - **every model is scored the same way:** same file, same units, same forecast times
  - **reproducible:** seeds fixed, test days never used for fitting, saved models reproduce their
    forecasts exactly
  - **tested:** 99 automated tests across data, models, metrics, simulator and policy

**Visual:** the pipeline flowchart, left to right, with the five layers as swim lanes.

**Script**

> "The whole system is one Python package with one command per stage. We download the trip files,
> preprocess them into five-minute counts per zone with the features, the flow graph and travel
> times, train three networks, and write every model's forecasts, ours and the baselines, into a
> single file. Then we calibrate, run the error analyses, and run the fleet simulation.
>
> Stages only talk to each other through files, so any stage can be rerun without redoing the
> others. A few decisions matter. Training is separate, because it needs a GPU; everything after
> runs on a laptop. Every model is scored through the same file in the same units, so no
> comparison depends on a special path. Seeds are fixed and the test days are never used for
> fitting. And there are ninety-nine automated tests."

---

## Slide 15: What we expect to do next

**Title:** Next steps

*These are directions we are considering. Keep only the ones you really intend to do.*

**On the slide**

| Direction | What it would add |
|---|---|
| **Turn the forecast into a service** | a live clock replaying trips, a forecast every 5 minutes, continuous scoring against baselines, drift monitoring, and a repositioning recommendation |
| **Isolate what each part contributes** | retrain without the graph; test adding plain adjacency and distance graphs; test each of the three graphs separately |
| **More data and wider tests** | more months and other seasons; another city |
| **Richer inputs** | weather, holidays and events |
| **Forecast ranges, not single numbers** | an uncertainty band the fleet can plan with |
| **Test the decision with real fleet data** | replace the synthetic fleet if positions become available |

**Visual:** a simple roadmap: "now: model and study" → "next: service and monitoring" → "later: more
data and cities".

**Script**

> "Where we go next. The most natural step is to turn the forecast into a service: replay trips
> through a live clock, forecast every five minutes, score ourselves against the baselines as the
> truth arrives, and watch for drift. Second, we want to isolate what each part contributes:
> retrain without the graph, and test whether plain adjacency or distance graphs add anything.
> Third, more months, other seasons and another city, with weather and events as inputs. And
> forecasts with ranges, not single numbers, so a fleet can plan around the uncertainty. With real
> fleet positions, we could also test the decision itself instead of a simulation."

---

## Questions to expect, and short answers

**On the temporal part**

| Question | Short answer |
|---|---|
| Why dilated convolutions and not an LSTM or GRU? | They read every step directly at several time scales, with few layers. Our first design used a GRU; the Graph WaveNet design is about 1 to 2% more accurate. That change altered the temporal and the spatial part together, so we cannot give the share due to the temporal part alone. |
| What does "causal" mean here? | An output at time t uses only steps at t or earlier; nothing from the future leaks in |
| Why filter length 2 and these dilations? | It is the standard Graph WaveNet pattern, repeated; the reach of 39 steps fits in the 48-step window. We did not tune it. |
| Why take only the newest time step for the skip? | After the dilated layers, the newest step is the only one that has seen the whole reach; the earlier ones have seen less |
| What does the gate do, in plain words? | It decides how much of the proposed signal gets through, so irrelevant history can be switched off |
| Why log and standardise the counts? | Counts are skewed and zones differ hugely in size; log and per-zone standardising put them on one scale |
| Why sine and cosine for time? | So the clock wraps around: 23:55 is next to 00:00 |
| What is batch normalisation for? | It keeps the numbers well scaled between layers so training is stable |

**On the whole model and the system**

| Question | Short answer |
|---|---|
| How are the three networks combined? | Their forecasts are averaged in pickup counts |
| How do you avoid overfitting? | Early stopping on validation loss, dropout 0.1 in the graph layers, a small weight penalty, shuffled windows, and a test period never used for fitting |
| What is the loss, and why Poisson? | The Poisson negative log-likelihood on counts, because pickups are counts and mostly small |
| How fast is a forecast? | About 70 milliseconds for all 258 zones on a laptop CPU (one window, measured); training took hours on a GPU |
| Could it run live? | The model could: it needs the last four hours of counts. We have not built the live pipeline; that is the first item under next steps |
| What is calibration? | A small per-zone correction fitted on validation days only, then applied to test forecasts; the strongest baseline gets the same one |
| Why not weather or events as inputs? | Not included in the shipped model; weather was tested early on and did not help the first design. Events are not in the data we used. |
| What is not in the architecture that you would add? | Forecast uncertainty, weather and events, and a retraining schedule |

---

## Numbers to have in your head

| Fact | Value |
|---|---|
| Input per zone per step | 7 numbers |
| Window | 48 steps = 4 hours |
| Filter length; dilations | 2; 1, 2, 4, 8, 16, 1, 2, 4 |
| Reach | 39 steps = 3 h 15 min |
| Time axis through the layers | 48, 47, 45, 41, 33, 17, 16, 14, 10 |
| Channels | 32 inside, 128 in the output layers |
| Layers | 8, each with a gated temporal convolution and a graph convolution |
| Output | 258 zones x 4 horizons |
| Extra head inputs | demand at the target time one day and one week earlier (with a flag for each) |
| Parameters | 147,644 (graph layers 39%, skips and output 34%, temporal 23%, rest 4%) |
| Training | Poisson loss; AdamW, learning rate 0.001; batch size 32; up to 45 epochs, stop after 6 without improvement; gradient clipping at 1.0; dropout 0.1 |
| Networks | 3, trained from different random starts, averaged |
| One forecast on a CPU | about 70 ms |
| Tests | 99 |
| Commands | download, preprocess, train, export, baselines, calibrate, align, headroom, regimes, simulate, plots |

## Before you present

- **Be able to draw slide 13 from memory,** with the shapes. It is the question that separates
  people who built the model from people who read about it.
- **Know the temporal ablation honestly:** we changed the temporal and spatial design at the same
  time, so there is no separate number for the temporal part.
- **Check the future-work table with your partner** and delete anything you do not intend to do.
- **Time it.** The scripts add up to about 815 words; read them aloud once with a timer.
