home-assistant-climate-tpi-thermostat
===

Home Assistant Custom Component adding a TPI Thermostat for Pilot Wired heaters connected using a Qubino ZMNHJD1
This is heavely inspired from : 
- https://github.com/piitaya/home-assistant-qubino-wire-pilot
- https://forum.hacf.fr/t/gestion-de-bout-en-bout-du-chauffage/4897



# Introduction



# Settings


Key | Type | Required | Description
platform | string | yes | Platform name
heater | string | yes | Light entity exposed by the Qubino
in_temperature_sensor | string | yes | Indoor temperature sensor measuring room temperature (also used for display)
out_temperature_sensor | string | yes | Temperature sensor (for display)
additional_modes	boolean	no	6-order support (add Comfort -1 and Comfort -2 preset)
name	string	no	Name to use in the frontend.
unique_id	string	no	An ID that uniquely identifies this cover group. If two climates have the same unique ID, Home Assistant will raise an error.
