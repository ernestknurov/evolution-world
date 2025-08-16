# Evolution World - Project Documentation

## Agent State and Interactions

The agent system is designed with a complex state management and interaction model that simulates realistic survival mechanics.

### Agent State Bars

The agent maintains four critical state bars that determine its survival and behavior:

#### 1. Food Level (Hunger)
- **Range:** 0 to 10
- **Function:** Indicates the agent's hunger level
- **Mechanics:**
  - Eating food increases `food_level` by `food_saturation`
  - Eating food also increases `energy_level` by `food_energy`
  - Food level decreases naturally each tick
  - **Starvation Effect:** When food level reaches zero:
    - Agent receives `food_starvation` penalty
    - Energy level decreases by `food_starvation_energy_lost` each tick

#### 2. Water Level (Thirst)
- **Range:** 0 to 10
- **Function:** Indicates the agent's thirst level
- **Mechanics:**
  - Drinking water increases `water_level` by `water_saturation`
  - Drinking water also provides small energy boost (`water_energy`)
  - Water level decreases naturally each tick
  - **Dehydration Effect:** When water level reaches zero:
    - Agent receives `water_starvation` penalty (larger than food penalty)
    - Energy level decreases by `water_starvation_energy_lost` (larger loss) each tick

#### 3. Energy Level
- **Range:** 0 to 100
- **Function:** Main power source for all agent actions
- **Mechanics:**
  - Every action requires energy expenditure
  - Agent can rest (do nothing) to gain small energy amounts
  - Can be replenished through eating food and drinking water
  - **Death Condition:** If energy level drops below zero, agent dies and receives death penalty

#### 4. Lifespan
- **Type:** Fixed value
- **Function:** Maximum time the agent can live
- **Mechanics:**
  - Predetermined and unchangeable throughout the game
  - Represents natural aging limit

### Survival Strategy

The system is designed with the principle that:
- **Food** provides energy and extends survival time
- **Water** is more critical for short-term survival
- **Energy management** is essential for all activities
- **Balance** between resource consumption and conservation is key to longevity

Current goal is to survive as long as possible. For this you need to find food and water as well as manage your energy levels effectively. The agent must learn to prioritize actions based on immediate needs (like drinking water) while also considering long-term sustainability (like eating food to maintain energy).

This creates a realistic survival simulation where agents must learn to balance immediate needs (water) with long-term sustainability (food and energy management).