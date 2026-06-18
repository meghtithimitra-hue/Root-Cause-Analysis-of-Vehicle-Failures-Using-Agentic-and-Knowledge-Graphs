# Automotive Fault Knowledge Graph — Schema

## Overview

A knowledge graph modeling automotive fault diagnosis. The graph encodes how vehicle **Systems** are composed of **Components**, which exhibit **Symptoms**, are diagnosed via **DiagnosticTests** that yield **Results**, and ultimately require **RepairActions**.

---

## Node Labels

| Label | Count | Description | Property |
|---|---|---|---|
| `System` | 13 | Major vehicle system (e.g., ABS, Engine, Transmission) | `name` |
| `Component` | 98 | Repairable sub-assembly or part within a system | `name` |
| `Symptom` | 143 | Observable problem reported by driver or technician | `name` |
| `DiagnosticTest` | 190 | Specific check or measurement step | `name` |
| `Result` | 290 | Possible outcome of a diagnostic test | `name` |
| `RepairAction` | 98 | Remedial action (derived as `"Replace <Component>"`) | `name` |

All nodes carry a single `name` property (unique per label).

---

## Relationships

| Relationship | Subject → Object | Cardinality | Description |
|---|---|---|---|
| `HAS_COMPONENT` | `System` → `Component` | 1:N | A system is made up of one or more components |
| `SHOWS_SYMPTOM` | `Component` → `Symptom` | 1:N | A faulty component exhibits certain symptoms |
| `DIAGNOSED_BY` | `Component` → `DiagnosticTest` | 1:N | A component can be diagnosed via specific tests |
| `HAS_RESULT` | `DiagnosticTest` → `Result` | 1:N | A diagnostic test produces possible outcomes |
| `REQUIRES_FIX` | `Component` → `RepairAction` | 1:1 | A fault in a component demands a repair action |

---

## Hierarchy

### ABS System (9 components)
ABS Control Module · ABS Wheel Speed Sensor · Brake Booster · Brake Caliper · Brake Hose · Brake Master Cylinder · Brake Pad · Brake Rotor · Brake Shoe & Drum

### Air Conditioning System (8)
AC Compressor · AC Condenser · AC Evaporator · AC Recharge · Air Conditioning Diagnosis · Cabin Air Filter · Heater Blower Motor · Heater Blower Motor Resistor

### Cooling System (8)
Coolant/Antifreeze · Coolant Leak Diagnosis · Engine Cooling System · Heater Core · Heater Hose · Radiator/Cooling Fan · Radiator Hose · Water Pump

### Drivetrain (11)
Bevel Gears · Center Differential · Clutch Cable · Clutch Slave Cylinder · Differential · Live Axle · Rigid Axle · Slushbox · Universal Joint · Viscous Coupling · Shaft and Tires

### Electrical System (14)
Battery Replacement · Charging System · Check Engine Light Diagnosis · Door Window Motor · Door Window Regulator · Headlamp Bulb · Ignition Distributor Cap · Ignition Wire Set · Power Door Lock Actuator · Starter Motor · Windshield Washer Pump · Wiper Motor · Horn · Battery Charging

### Emissions System (9)
Canister Purge Valve · Catalytic Converter · Charcoal Canister · Exhaust Manifold · Exhaust Manifold Gasket · Fuel Tank Pressure Sensor · Knock Sensor · M.A.P. Sensor · Oxygen Sensor

### Engine Components (15)
Balance Shaft · Boost Pressure · Diesel Injection Pump · Diesel Glow Plug · Engine Mounts · Exhaust Valve · Fuel Injector · Idle Air Control Valve · Intake Valve · Oil Pump · Piston Rings · Timing Belt · Valve Cover Gasket · Fan Belt · Engine Belt

### Steering (1)
Steering Noise

### Engine Compartment (6)
Air Filter · Engine Control Unit (ECU) · Engine Oil · Fuel Filter · PCV Valve · Throttle Body

### Fuel System (5)
Fuel Injector · Fuel Pressure Regulator · Fuel Pump · Fuel Tank · Throttle Position Sensor

### Liquid Systems (5)
Brake Fluid · Coolant Reservoir · Power Steering Fluid · Radiator · Windshield Washer Fluid

### Transmission (5)
Clutch Master Cylinder · Torque Converter · Transmission Fluid · Transmission Filter · Transmission Solenoid

### Wheels & Tires (3)
Wheel Bearing · Tire Pressure Monitoring System (TPMS) · Wheel Hub

---

## Example Subgraph (ABS System)

```
(:System {name: "ABS System"})
    │
    │ HAS_COMPONENT
    ▼
(:Component {name: "ABS Control Module"})
    │
    ├── SHOWS_SYMPTOM ──→ (:Symptom {name: "ABS warning light on"})
    ├── SHOWS_SYMPTOM ──→ (:Symptom {name: "Brake pedal pulsation"})
    │
    ├── DIAGNOSED_BY ──→ (:DiagnosticTest {name: "Check ABS fuse"})
    │                        │
    │                        ├── HAS_RESULT ──→ (:Result {name: "Blown"})
    │                        └── HAS_RESULT ──→ (:Result {name: "Intact"})
    │
    ├── DIAGNOSED_BY ──→ (:DiagnosticTest {name: "Inspect wiring to ABS module"})
    │                        │
    │                        ├── HAS_RESULT ──→ (:Result {name: "Faulty wiring"})
    │                        └── HAS_RESULT ──→ (:Result {name: "Good wiring"})
    │
    └── REQUIRES_FIX ──→ (:RepairAction {name: "Replace ABS Control Module"})
```

---

## Graph Statistics

| Metric | Value |
|---|---|
| Total nodes | 832 |
| Total relationships | 988 |
| Systems | 13 |
| Components | 98 |
| Symptoms | 143 |
| DiagnosticTests | 190 |
| Results | 290 |
| RepairActions | 98 |
| Relationship types | 5 |
| Node labels | 6 |

---

## Data Source

Derived from `automotive_faults_aktc_obike_et_al.json`.  
Ingestion pipeline: `convert_to_triples.py` → `triples.csv` → `ingest_neo4j.py`.  
