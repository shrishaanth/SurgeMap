# Presentation: Member 1's part

Slides 1 to 7 of the 15-minute talk: the real-world use case, the problem, how it is solved
classically and why that falls short, our approach, and the graph. Member 2 continues from the
temporal part onward.

**Time budget: about 7.5 minutes.** The scripts total roughly 990 words, which is about 7.6
minutes at a spoken pace of 130 words a minute. Replace `[Name 1]` and `[Name 2]` with your names.

| # | Slide | Minutes |
|---|-------|---------|
| 1 | The real-world use case | 1.2 |
| 2 | The problem statement | 1.1 |
| 3 | How it is usually solved, without machine learning | 1.1 |
| 4 | The problems they face | 1.3 |
| 5 | Our approach | 1.0 |
| 6 | Why zones need each other | 0.65 |
| 7 | Building the graph | 1.3 |
| | **Total** | **7.65** |

---

## Slide 1: The real-world use case

**Title:** Where will demand appear next?
**Subtitle:** Forecasting taxi pickups, zone by zone, up to an hour ahead

**On the slide**

*The situation*
- Idle vehicles sit in one area while riders wait in another
- Riders wait or give up; drivers burn fuel driving empty
- A fleet can only fix this **ahead of time**

*Who uses the forecast, and for what*

| Who | The decision it drives |
|---|---|
| Fleet operators, dispatchers | which idle vehicles to move, and where |
| Ride-hailing platforms | driver incentives and pricing in busy areas |
| Drivers | which area to head towards |
| City and airport authorities | taxi-stand capacity and event-day planning |

*In industry*
- Ride-hailing platforms forecast demand and supply for small map areas in short time steps. Uber
  uses it to direct drivers to demand before it appears; Lyft feeds it into pricing and driver
  incentives (1, 2)

*This project*
- The same question, on public New York taxi data, with the forecast tested on a simulated fleet

**Footnote:** (1) Uber Engineering, "Forecasting Introduction". (2) Lyft Engineering, "Real-Time
Spatial Temporal Forecasting @ Lyft". Links are at the end of this document.

**Visual:** a city map with a cluster of idle vehicles in one area, a hot spot of waiting riders in
another, and an arrow between them. Both presenters' names on the slide.

**Script**

> "Imagine a taxi fleet at six on a Friday evening. In one part of the city, a dozen vehicles are
> sitting idle. In another, riders are opening their apps and nobody is close. Riders wait, or give
> up, and the vehicles burn fuel driving empty. The fleet could have fixed this if it had moved
> earlier, but moving earlier needs one thing: knowing where demand is about to appear.
>
> That is a forecasting question, and it is the one the big platforms work on. Uber describes
> forecasting rider demand and driver supply at fine spatial and time detail, to direct drivers to
> demand before it arises. Lyft forecasts for small map cells in five-minute steps, an hour ahead,
> and feeds that into pricing and driver incentives.
>
> We take the same question and ask it with public data: New York's yellow-taxi trips. And we do
> not stop at forecasting accuracy: we also test what the forecast is worth to a fleet."

*[Name 2] adds one line:* "I'll cover the model, the results, and the system around it."

---

## Slide 2: The problem statement

**Title:** The problem, stated precisely

**On the slide**

| | |
|---|---|
| **Given** | pickups in each of 258 NYC taxi zones for each of the last 48 five-minute intervals (4 hours), plus time of day, day of week and dropoffs |
| **Predict** | pickups in every zone **5, 15, 30 and 60 minutes ahead** |
| **Data** | about 13 million yellow-taxi trips, October 2023 to January 2024 |
| **Tested on** | 27 to 31 January 2024, days never used for fitting |
| **Success** | lower error than standard forecasting methods, **and** a better decision when the forecast drives a fleet |

*What makes it hard*
- **Mostly zeros:** 80% of zone-and-time cells have no pickup, and arrivals are random
- **Time and space together:** a zone depends on its own past, the clock, and other zones
- **Very uneven:** the busiest 30 of 258 zones carry 81% of all pickups

**Visual:** one small chart: a busy zone and a quiet zone over 24 hours.

**Script**

> "Stated precisely: for every one of New York's 258 taxi zones, we give the model the last four
> hours of pickups, in five-minute steps, plus the time of day and day of the week. We ask for the
> number of pickups in each zone five, fifteen, thirty and sixty minutes ahead.
>
> We train on four months, about thirteen million trips, and test on five days the model never saw.
>
> Two things make it hard. First, most of the data is zeros: in four out of five zone-and-time
> cells nobody was picked up, and pickups arrive at random. Second, a zone's next hour depends on
> its own history, on the clock, and on other zones, all at once.
>
> We judge success twice: does it forecast better than standard methods, and does a fleet make
> better decisions when it uses the forecast."

---

## Slide 3: How it is usually solved, without machine learning

**Title:** Classical forecasting: write the pattern down as an equation

**On the slide**

Let *y(t)* be pickups in one zone in time bin *t*, and *ŷ(t+h)* the forecast *h* bins ahead.

| Method | The idea | The equation |
|---|---|---|
| **Naive** | the next value equals the last | ŷ(t+h) = y(t) |
| **Seasonal average** | average past values at this time of day and weekday | ŷ = mean of y(s) over earlier bins in the same slot |
| **Exponential smoothing** | a running level; recent values count more | ℓ(t) = α·y(t) + (1 − α)·ℓ(t−1) |
| **Autoregression** | a weighted sum of the last *p* values | y(t) = c + Σ φᵢ·y(t−i) + ε |
| **ARIMA** | autoregression + differencing + past errors | adds Σ θⱼ·ε(t−j) to the above |
| **Poisson arrivals** | random arrivals at a rate that follows the clock | P(y = k) = λᵏ · e^(−λ) / k! |

*Bringing in space*

| Method | The idea | The equation |
|---|---|---|
| **Vector autoregression / STARIMA** | each zone depends linearly on others through a fixed neighbour matrix **W** | y(t) = c + Σ Φᵢ·W·y(t−i) + ε |
| **Gravity model** | flow between two zones from their size and distance | F(ij) = k · mᵢ·mⱼ / d(ij)^β |

**Footer:** In this project we keep three of these as baselines: naive, seasonal average, and
autoregression (as ridge regression).

**Script**

> "Before machine learning, forecasting was done by writing the pattern down as an equation and
> fitting a few numbers. The simplest is naive: tomorrow looks like today. Next, the seasonal
> average: take the same time of day and weekday from the past. Exponential smoothing keeps a
> running level. Autoregression and its extension ARIMA predict from weighted sums of recent
> values. And the Poisson model treats pickups as random arrivals at a clock-driven rate.
>
> Almost all of these look at one zone at a time. Two classical ways bring in other zones: vector
> autoregression, where each zone depends linearly on the others through a neighbour matrix
> someone writes down, and the gravity model from transport planning.
>
> We keep three of these as baselines, so we can measure the neural network against the classical
> approach on the same data."

---

## Slide 4: The problems they face

**Title:** Where classical methods fall short

**On the slide**

| Problem | Why it hurts here |
|---|---|
| **Linear** | every model is a weighted sum; it cannot say "a rise next door matters only late on a weekend night" |
| **Structure chosen by hand** | the number of lags, which season, which zones are neighbours are fixed before fitting; the model cannot discover one that was not written in |
| **One zone, or a fixed neighbour matrix** | vector autoregression across 258 zones needs about 66,000 coefficients per lag, and the neighbours are still our guess |
| **Long, layered seasons** | one day is 288 bins and one week is 2,016; classical seasonal models are built for short seasons, one at a time |
| **Wrong noise model for counts** | most assume smooth, symmetric errors; here 80% of cells are zero |
| **Blind to the present, or static** | the seasonal average cannot see today's surge; the gravity model describes typical flow, not the next 15 minutes |

*What we measured in this project (RMSE, 5 minutes to 60 minutes ahead)*
- **Naive:** 1.72 → 2.19 (error grows 28% as the horizon lengthens)
- **Seasonal average:** 1.54 → 1.55 (flat: it knows the routine, never the present)
- **Autoregression (ridge):** 1.43 → 1.77 (grows 24%)

*So a better method must be:* **non-linear · able to learn its own structure · time and space in one
model · count-aware**

**Visual (optional):** a one-day chart of one zone's actual pickups against its naive and
seasonal-average forecasts.

**Script**

> "These methods have a few problems that matter here. They are linear: each is a weighted sum,
> so a pattern like 'a rise next door only matters late on weekend nights' cannot be expressed.
> Their structure is chosen by a person: how many lags, which season, which zones count as
> neighbours. The model cannot discover a relationship that was not written in.
>
> Space is handled badly: either one zone at a time, or a vector autoregression with about
> sixty-six thousand coefficients per lag for 258 zones, and a neighbour matrix we still have to
> guess. And the noise is wrong: these models assume smooth errors, while four out of five cells
> here are zero.
>
> You can see it in our own numbers. Naive gets steadily worse with the horizon, 28% worse by an
> hour. The seasonal average is flat: it knows the routine but not what is happening now.
> Autoregression degrades by a quarter.
>
> So we need something non-linear, that learns its own structure, handles time and space
> together, and respects counts."

---

## Slide 5: Our approach

**Title:** One network that learns time and space together

**On the slide**
- **The idea:** model the city as a **graph**, and let a single neural network learn each zone's
  pattern over time *and* the relationships between zones
- **Three ingredients**
  1. **The graph:** zones are nodes, trip flow gives the connections
  2. **The temporal part:** each zone's last 4 hours, read at several time scales
  3. **The spatial part:** zones exchange what they have learned, over the graph
- **Model:** Graph WaveNet (Wu et al., 2019); our forecast is the average of three networks

*It answers the four problems from the last slide*

| Problem | How the approach answers it |
|---|---|
| Linear | a neural network learns non-linear patterns |
| Structure chosen by hand | filters and part of the graph are learned from data |
| Time and space apart | both live in one model, layer after layer |
| Wrong noise for counts | trained with a Poisson loss, the right model for counts |

**Visual:** a three-box diagram (Graph, Temporal, Spatial) with arrows into one forecast box.

**Script**

> "Our approach answers those problems directly. We treat the city as a graph: each zone is a
> node, and the connections say which zones tell us something about each other. One neural
> network then learns two things together: how each zone's demand moves over time, and how zones
> influence one another.
>
> Being a neural network, it is non-linear. Its filters and part of its graph are learned from
> data, not written by hand. Time and space are in one model. And we train it with a Poisson
> loss, which is the right noise model for counts.
>
> The specific design is called Graph WaveNet, and our forecast averages three of them. I'll cover
> the graph; my partner will cover how the network reads time and how the two are combined."

---

## Slide 6: Why zones need each other

**Title:** Demand in one zone is a clue about another

**On the slide**
- Zones are not independent. Passengers dropped in one place become demand somewhere nearby, a
  little later.
- **Patterns the spatial part is there to capture**

| Pattern | Example |
|---|---|
| Demand that travels | arrivals at an airport become pickups in the hotel districts afterwards |
| Spillover | when one zone is saturated, riders walk to the next |
| Zones that move together | two business districts far apart both peak at 18:00 |
| Events wider than one zone | rain or a public event shifts a whole area |

- **A grid of squares does not fit:** zones are irregular shapes, and related zones are often far
  apart on the map

**Visual:** a map with two arrows: "airport to city", and "two distant business districts, same
curve".

**Script**

> "Why space? Because zones are not independent. A passenger dropped off in one place becomes
> demand somewhere nearby shortly afterwards. When one zone saturates, riders spill to the next.
> Two business districts on opposite sides of the city peak together at six. And some events, such
> as rain or a concert, shift a whole area at once.
>
> These relationships do not follow a square grid. Zones are irregular, and related zones are often
> far apart, so a graph is the natural structure."

---

## Slide 7: Building the graph

**Title:** Three graphs: where trips go, where they come from, and what the model finds itself

**On the slide**
- **Nodes:** the 258 taxi zones
- **Edges:** from the trip data, in two directions, plus one the model learns

| Graph | What it says | How it is built |
|---|---|---|
| **Outgoing flow** | of the trips leaving zone *i*, the share that go to zone *j* | count trips *i → j* over the data, divide each row by its total |
| **Incoming flow** | of the trips arriving at zone *i*, the share that came from *j* | the same counts, reversed |
| **Learned adjacency** | which zones behave alike, wherever they are | two 10-number vectors per zone, learned in training; scored and normalised so each zone gets a distribution over all others |

- **Flow, not map distance:** two distant zones with heavy traffic count as close; neighbours that
  exchange few trips count as far
- **Direction matters:** airport-to-city is not city-to-airport
- 37,475 of the 66,564 possible zone pairs carry at least one trip

**Visual:** a map with the strongest flows drawn as arrows, beside a small heat map of the flow
matrix.

**Script**

> "The graph has the 258 zones as nodes. For the connections, we use the trip data itself. Count
> how many trips go from each zone to each other zone, and divide by the zone's total: that gives
> the share of its trips going to each destination. We keep it directional, so airport-to-city
> differs from city-to-airport, and we build the incoming version the same way, reversed.
>
> We use trip flow, not map distance, because it measures how people actually move. Two zones far
> apart with heavy traffic count as close; two neighbours with few trips between them do not.
>
> The third graph is not given. Each zone has two small vectors of ten numbers that the network
> learns. From them it works out which zones are useful to each other for forecasting, including
> similar districts with no trips between them.
>
> So each zone has three views of its surroundings. [Name 2] will now show how the network reads
> time, and how these graphs are used inside it."

**Handoff to Member 2:** the graph slide covers *what the graph is*. How information is passed over
it (the graph convolution) is in Member 2's slide on how the parts are combined.

---

## Questions to expect, and short answers

**On the use case and problem**

| Question | Short answer |
|---|---|
| Why taxi data, and why New York? | It is public, official, large and zone-based, so it stands in for any fleet that works by zones |
| Why 5-minute bins and up to 60 minutes? | Five minutes is fine enough to see a surge build; fifteen is the interval our repositioning policy decides at; sixty is about how far ahead moving a vehicle can plausibly pay off |
| Is this deployed anywhere? | No. Platforms run systems like this internally; ours is a study on public data with a simulated fleet |
| Why not just use last week's average? | That is one of our baselines (the historical average); it knows the routine but not what is happening now, and we measure how much the model adds |
| Does your model use live data? | No. It was trained and tested on historical files; a live system would feed it the last four hours as they arrive |

**On the classical methods**

| Question | Short answer |
|---|---|
| Did you try ARIMA? | No. Ridge regression on the last 48 bins covers the autoregressive idea; ARIMA mostly adds differencing and error terms. We did not run it, so we do not report it. |
| Is the seasonal average really "not ML"? | It is plain statistics, a mean per time slot. Gradient boosting is the one baseline we include that is machine learning, as the standard strong competitor |
| Don't companies still use classical models? | Yes, often as part of a larger system. Our result is that the network adds a few percent over a tuned boosted model, and more in surges |
| Why is the seasonal average flat across horizons? | It ignores the current state entirely, so its error is the same whether you ask for 5 or 60 minutes |

**On the approach and the graph**

| Question | Short answer |
|---|---|
| Why a graph and not a grid image (a CNN)? | Zones are irregular, and related zones are often far apart; an image filter only sees neighbours |
| Why flow and not distance or adjacency? | Flow measures actual movement and already carries most of what adjacency would, since many trips are short. We did not test adding adjacency or distance graphs; the learned graph can pick up what flow misses |
| Is the graph built using the test days? | Yes. The flow counts use all trips in the date range, which includes the five test days (about 4% of the data). This is a mild form of leakage, and it is listed as a limitation. |
| What does the learned graph actually find? | We have not inspected it; it should not be read as a map or as causation |
| Does the graph really help? | In our first design it was worth about 1% or less; for the shipped network we did not isolate it, because the whole architecture changed at once |

---

## Numbers to have in your head

| Fact | Value |
|---|---|
| Trips | about 13 million (13,203,191 rows; 13,037,653 pass validity checks) |
| Zones | 258 |
| Time bins | 5 minutes; 35,424 in the four months |
| Input window | 48 bins (4 hours) |
| Horizons | 5, 15, 30, 60 minutes (1, 3, 6, 12 bins) |
| Test window | 27 to 31 January 2024, 1,329 forecast times |
| Cells with no pickup | 80% |
| Share of pickups in the busiest 30 zones | 81% |
| Naive error, 5 to 60 minutes | 1.716 to 2.189 |
| Seasonal average error | 1.544 to 1.553 |
| Ridge (autoregression) error | 1.428 to 1.769 |
| Possible zone pairs, 258 x 258 | 66,564 |
| Zone pairs with at least one trip | 37,475 |
| Learned adjacency | two 10-number vectors per zone |

## Before you present

- **Open the Uber and Lyft posts once** (links below) and check that slide 1's wording is a fair
  reading. An evaluator may look them up.
- **Say "ARIMA was not run" before you are asked.** Ridge regression is the autoregression baseline.
- **Time it.** The scripts add up to about 990 words; read them aloud once with a timer. If you
  run over, cut slide 3's equations down to naive and vector autoregression, and drop the
  "what we measured" paragraph on slide 4 to one sentence.
- **Know the graph limitation cold:** flow built from all trips, including the test days.

## Sources

- Uber Engineering: [Forecasting Introduction](https://www.uber.com/gb/en/blog/forecasting-introduction/)
- Lyft Engineering: [Real-Time Spatial Temporal Forecasting @ Lyft](https://eng.lyft.com/real-time-spatial-temporal-forecasting-lyft-fa90b3f3ec24)
- Wu et al., 2019: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
