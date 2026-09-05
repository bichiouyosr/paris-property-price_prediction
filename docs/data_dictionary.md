# Data dictionary

Two tables: the 35 columns of the raw extract as delivered, and the columns
built on top of them. `scripts/make_synthetic_data.py` reproduces the raw
schema exactly, so this file also documents what the generator emits.

## Raw extract (35 columns)

| Column | Type | Notes |
|---|---|---|
| `Unnamed: 0` | int | Spare row index from the export. Dropped. |
| `external_id` | int | Listing identifier. Used to detect relistings, then dropped. |
| `client_id` | int | Agency identifier. Not a property attribute — dropped. |
| `address_id` | int | Address identifier. Used to repair zipcodes, then dropped. |
| `street_id` | int | Street identifier. Same use, same fate. |
| `iris_id` | int | INSEE statistical block. High cardinality, dropped. |
| `parcel_id` | — | 100% missing. Dropped. |
| `city` | str | Free text, inconsistently spelled ("PARIS", "Paris Cedex", " Paris"). Superseded by `zipcode`. |
| `city_id` | int | Constant (75056 = Paris). Dropped as near-constant. |
| `zipcode` | int | 75001–75020 when valid. ~1% are not. Repaired, see below. |
| `price` | float | **Target.** Asking price in EUR. |
| `area` | float | Habitable surface, sqm. The dominant predictor. |
| `floor` | float | Floor of the unit. Negative = basement level. |
| `floor_count` | float | Floors in the building. |
| `room_count` | float | Rooms (French "pièces": living rooms + bedrooms, not kitchen/bath). |
| `bedroom_count` | float | Bedrooms. |
| `bathroom_count` | float | Bathrooms. |
| `toilet_count` | float | Separate WCs. |
| `balcony_count` | float | Missing means none, not unknown. |
| `terrace_count` | float | Missing means none, not unknown. |
| `balcony_area` | float | sqm. |
| `terrace_area` | float | sqm. |
| `parking_boxe_count` | float | Enclosed parking boxes. |
| `parking_inside_count` | float | Parking spaces inside the building. |
| `parking_garage_count` | — | >99% missing. Dropped. |
| `site_area` | — | >99% missing. Dropped. |
| `has_cellar` | float | 0/1. Missing treated as 0. |
| `has_passenger_lift` | float | 0/1. Missing treated as 0. |
| `build_year` | float | ~53% missing; contains 0 and 3021. |
| `item_type` | str | `ITEM_TYPE.APARTMENT`, `ITEM_TYPE.HOUSE`, plus parking/office/storage rows to filter out. |
| `item_subtype` | str | Same prefix; flat, studio, duplex, loft, villa… |
| `energy_certificate_efficiency_score` | str | `ENERGY.CERTIFICATE_EFFICIENCY_SCORE.A` … `.G`. ~48% missing. |
| `start_date` | datetime | Listing start. |
| `end_date` | datetime | Listing end. |
| `publish_start_date` | datetime | First publication. |

### Known zipcode defects

| Value | Reading | Treatment |
|---|---|---|
| 75116 | Legacy postal code for the 16th | → 75016 |
| 75107, 75108 | Typos | → 75007, 75008 |
| 75000 | Generic "Paris" placeholder | → 75003 |
| 75550, 75023 | Not real Paris codes, no obvious target | Inferred from other listings at the same `address_id` + `street_id`; otherwise NaN |

## Engineered features

| Feature | Built from | Why it exists |
|---|---|---|
| `arrondissement` | `zipcode - 75000` | 1–20, usable as an ordered value |
| `rive_gauche` | arrondissement ∈ {5,6,7,13,14,15} | Left Bank carries a price premium the zipcode number does not encode |
| `prime_arrondissement` | arrondissement ∈ {6,7,8,16} | The four highest EUR/sqm districts |
| `property_age` | `listing_year - build_year`, clipped at 0 | Age matters more than the year itself |
| `area_per_room` | `area / room_count` | Layout: 90 sqm across 2 rooms ≠ 90 sqm across 4 |
| `area_per_bedroom` | `area / bedroom_count` | Same idea, sleeping capacity |
| `outdoor_area` | `terrace_area + balcony_area` | Outdoor space priced as one thing |
| `total_parking` | box + inside counts | Parking priced as one thing |
| `floor_ratio` | `floor / floor_count` | Top-floor premium is relative, not absolute |
| `energy_score_ordinal` | DPE letter → 7 (A) … 1 (G) | Ordered, so a model can use the direction |
| `listing_month`, `listing_quarter`, `listing_dow` | `publish_start_date` | Seasonality of the listing calendar |
| `type_*`, `subtype_*` | one-hot of type/subtype | Property category |

## Built but excluded from the model

| Column | Why it is excluded |
|---|---|
| `price_per_sqm` | Derived from the target — direct leakage |
| `log_price` | Derived from the target — direct leakage |
| `log_area` | Collinear with `area`; kept for EDA only |
| `days_on_market` | Only known once the listing closes; unavailable at prediction time |
