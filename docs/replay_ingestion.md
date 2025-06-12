# Replay Ingestion

Replay ingestion plays an important part in the repo, as it is what extract features from the replay, converting the SC2Replay file into a tabular form that can be analysed, processed, and eventually fed into a machine learning model. The SC2Replay files follow and event structure, where each 'action' that happens in game is represented by an event. The general approach to ingesting the replays is to ingest the events one by one, building up a picture of the game in the form of rows in a pandas dataframe. The diagrams below show this in great detail. All replay ingestion code can be found in the `src/starcraft_predictor/replays/` folder.

## Process Flow

```mermaid
graph TD
    A[Start] --> B[Load Replay]
    B --> C[Validate Replay]
    C --> D[Initialize Tracking]
    D --> E[Process Events]
    E --> F[Generate DataFrame]
    F --> G[End]
    
    subgraph Event Processing
        E --> E1[Unit Events]
        E --> E2[Player Stats Events]
        E --> E3[Upgrade Events]
    end
```

This diagram illustrates the main workflow of the replay ingestion process. It shows how a replay file moves through the system from initial loading to final DataFrame generation. The process begins with loading the replay file, followed by validation to ensure it matches the expected matchup type. The system then initializes tracking components for units and player stats. The core of the process is the event processing stage, which handles three main types of events: unit events (birth, death, type changes), player stats events (collected every 10 seconds), and upgrade events. Finally, the processed data is compiled into a DataFrame that represents the game state at regular intervals.

## Sequence Diagram

```mermaid
sequenceDiagram
    participant RI as ReplayIngester
    participant UT as UnitTracker
    participant PST as PlayerStatsTracker
    participant DF as DataFrame
    
    RI->>RI: ingest_replay()
    RI->>RI: init_replay_tracking()
    loop For each event
        RI->>RI: Check event type
        alt Unit Event
            RI->>UT: Process unit event
        else Player Stats Event
            RI->>PST: Process stats event
            PST-->>RI: Return stats data
            RI->>DF: Generate new row
        end
    end
```

This sequence diagram demonstrates the temporal flow of event processing in the system. It shows how events are handled sequentially, with the ReplayIngester acting as the coordinator. The process begins with replay ingestion and initialization. Then, for each event in the replay, the system checks the event type and routes it to the appropriate handler. Unit events are processed by the UnitTracker, while player stats events trigger the PlayerStatsTracker and result in new rows being added to the DataFrame. This diagram emphasizes the event-driven nature of the system and how different components interact over time.

## Data Transformation Flow

```mermaid
graph LR
    A[Raw Replay Events] --> B[Event Processing]
    B --> C[Unit State]
    B --> D[Player Stats]
    C --> E[DataFrame Row]
    D --> E
    E --> F[Final DataFrame]
    
    subgraph Data Processing
        C --> C1[Unit Counts]
        C --> C2[Unit Types]
        C --> C3[Upgrade Counts]
        D --> D1[Resources]
        D --> D2[Army Value]
    end
```

This diagram illustrates how raw replay data is transformed into a structured DataFrame. It shows the parallel processing of unit state and player statistics, which are then combined into individual DataFrame rows. The unit state includes information about unit counts and types, while player stats track resources and army value. These different data streams are merged to create a comprehensive view of the game state at each time interval. The final DataFrame contains all this information in a format suitable for analysis and prediction.

