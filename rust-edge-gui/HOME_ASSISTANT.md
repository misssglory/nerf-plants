# Home Assistant / FlowerCare timeseries

rust-edge-gui can poll numeric Home Assistant entities and place them in the same generic plot-window system as tracked plant/leaf mask area.

## Connect

1. In Home Assistant open your user profile and create/copy a **Long-Lived Access Token**.
2. In rust-edge-gui open **Home Assistant / FlowerCare timeseries**.
3. Set the base URL, for example `http://homeassistant.local:8123` or `http://192.168.1.20:8123`.
4. Paste the token. Enable **Remember token in local state.json** only if storing the token on this computer is acceptable.

## Add FlowerCare sensors

Open **Developer Tools → States** in Home Assistant and find the exact entities created by your FlowerCare integration/device. Add the numeric sensor entities you want to plot. Typical measurements are temperature, moisture, conductivity/EC, illuminance and battery, but entity IDs are installation-specific.

- Leave **attribute** empty to read the entity's primary `state`.
- If a numeric measurement is stored inside an entity attribute, enter that exact attribute name.
- Non-numeric states such as `unknown`, `unavailable`, `on` or `off` are skipped.

Use **Fetch now** for a current sample, **Load history** to backfill the selected number of hours, or **Auto poll** for ongoing snapshots.

## Put sensors on plots

Open **Timeseries / plot windows**:

- **New plot window** creates an independent floating chart.
- Edit a series **name** and **group** inline.
- Use the plot dropdown on a series to move only that series.
- Use **Move whole group** to move all series with the same group name at once.

A useful setup is to name/group related readings as `FlowerCare` and put them in their own plot. Shape area (`px`) and sensor measurements can technically share a plot, but separate plots are often easier to read because they use different physical units.

## Shape grouping with the image

In the **Processed** image hold **Shift** and drag a rectangle around pivot markers. The selected tracked-shape area series are assigned to a new `selection-N` dashboard group. This does not alter the permanent collision identity used by plant/leaf tracking.

## API endpoints used

- `GET /api/states/<entity_id>` for snapshots.
- `GET /api/history/period/<timestamp>?filter_entity_id=...&end_time=...` for backfill.
- `Authorization: Bearer <token>` for authentication.

## Sequence-local persistence

When an image sequence is active, configured Home Assistant entity definitions and all collected numeric samples are also written to `<sequence>/sequence_state.json`. Timeseries names, logical groups, visibility, target plot IDs, and plot-window state are stored there as well. Reopening that sequence restores the sensor history and dashboard layout automatically.

The Home Assistant base URL and access token are **not** copied into the sequence folder. Credentials remain in the app-local state/config only.
