/*
This file manages all the user interface (UI) logic
It defines a global ui object that acts as a bridge between the HTML controls (sliders, buttons, checkboxes) and the main application state (the config object in bloomscape.js)
It sets up event listeners to detect user input and uses a system of callbacks to notify the main application when actions like "regenerate" need to occur
*/
const ui = {
  // placeholders for callback functions, the main bloomscape.js script will assign a function to this properties
  onRegenerate: null,
  onPresetChange: null,
  onWater: null,

  // Syncronize the UI with the 'config' state
  updateControlsFromConfig: function () {
    // Synchronize the visual state of all HTML controls with the current state of the global config object
    // This is essential for ensuring the UI always reflects the actual parameters being used
    if (!window.config) return;

    const presetSelect = document.getElementById("presetSelect");  // gets a reference to the <select> HTML element for the presets dropdown by its ID
    if (presetSelect.options.length !== window.config.lSystem.presets.length) {  // if the current number of options is different from the number of presets defined in the config object
      presetSelect.innerHTML = '';                                               // the options need updating so this line completely clears the current content of the dropdown menu
      window.config.lSystem.presets.forEach((p, i) => {                          // iterates over each preset object (p) and its index (i) in the config.lSystem.presets array
        const option = document.createElement('option');                         // creates a new <option> HTML element for each preset
        option.value = i;                                                        // sets the value attribute of the option to the preset's index
        option.textContent = p.name;                                             // sets the visible text of the option to the preset's name
        presetSelect.appendChild(option);                                        // appends the newly created option to the presetSelect dropdown menu
      });
    }
    // Set the value of each element of the UI to match the corresponding value from the configuration
    presetSelect.value = window.config.lSystem.currentPreset;
    document.getElementById("angle").value = window.config.lSystem.angle;
    document.getElementById("iterations").value = window.config.lSystem.iterations;
    document.getElementById("growthSpeed").value = window.config.animation.growthSpeed;
    document.getElementById("autoCycle").checked = window.config.environment.autoCycle;
    document.getElementById("rainCheckbox").checked = window.config.weather.isRaining;
    document.getElementById("cloudsCheckbox").checked = window.config.weather.showClouds;
    document.getElementById("sunRaysCheckbox").checked = window.config.weather.showSunRays;
      
    const timeSlider = document.getElementById("dayTimeSlider");                    // gets a reference to the "Time of Day" slider element
    if (timeSlider) {
      timeSlider.value = window.config.environment.dayTime;                         // sets the slider's value to match the current dayTime from the config
      timeSlider.disabled = window.config.environment.autoCycle;                    // disables the slider (makes it non-interactive) if the autoCycle is active
      timeSlider.style.opacity = window.config.environment.autoCycle ? 0.5 : 1.0;   // provides visual feedback by making the slider semi-transparent when it is disabled
    }
  }
};

function uiInit() {
  // This function is called once when the application starts
  // It finds all the interactive HTML elements in the control panel and attaches the necessary event listeners to them
  // These listeners update the config object in response to user actions and trigger the appropriate callbacks

  // Preset
  const presetSelect = document.getElementById("presetSelect");
  presetSelect.addEventListener("change", () => {
    window.config.lSystem.currentPreset = parseInt(presetSelect.value, 10);  // updates the currentPreset in the global config object with the new value from the dropdown
    if (typeof ui.onPresetChange === "function") ui.onPresetChange();        // checks if the onPresetChange callback has been assigned a function, and if so, calls it
  });

  // Growth speed
  const growthSpeedEl = document.getElementById("growthSpeed");
  growthSpeedEl.addEventListener("input", () => {
    window.config.animation.growthSpeed = parseFloat(growthSpeedEl.value);  // updates the growthSpeed in the global config object with the slider's new value
  });

  // Day/Night cycle
  const autoCycleEl = document.getElementById("autoCycle");
  autoCycleEl.addEventListener("change", () => {
    window.config.environment.autoCycle = autoCycleEl.checked;             // updates the autoCycle boolean in the config object based on the checkbox's state
    ui.updateControlsFromConfig();                                         // calls updateControlsFromConfig (see line 13) to reflect the change (e.g., disabling the manual time slider)
  });

  const dayTimeSlider = document.getElementById("dayTimeSlider");
  dayTimeSlider.addEventListener("input", () => {
    if (!window.config.environment.autoCycle) {                             // checks if the auto cycle is disabled (the slider should only have an effect in manual mode)
      window.config.environment.dayTime = parseFloat(dayTimeSlider.value);  // updates the dayTime in the config object if in manual mode
    }
  });

  // EDITING
  // Angle
  const angleEl = document.getElementById("angle");
  angleEl.addEventListener("input", () => {
    window.config.lSystem.angle = parseFloat(angleEl.value);              // updates the angle value in the config object
  });

  // Iterations
  const iterationsEl = document.getElementById("iterations");
  iterationsEl.addEventListener("input", () => {
    window.config.lSystem.iterations = parseInt(iterationsEl.value, 10);  // updates the iterations value in the config object
    if (typeof ui.onRegenerate === "function") {                          // if the onRegenerate callback exists
      ui.onRegenerate();                                                  // calls it to automatically regenerate the plant when its complexity changes
    }
  });

  // WEATHER
  const rainCheckboxEl = document.getElementById("rainCheckbox");
  rainCheckboxEl.addEventListener("change", () => {
    window.config.weather.isRaining = rainCheckboxEl.checked;             // updates the rain checkbox value in the config object
  });

  const cloudsCheckboxEl = document.getElementById("cloudsCheckbox");
  cloudsCheckboxEl.addEventListener("change", () => {
    window.config.weather.showClouds = cloudsCheckboxEl.checked;         // updates the clouds checkbox value in the config object
  });

  const sunRaysCheckboxEl = document.getElementById("sunRaysCheckbox");
  sunRaysCheckboxEl.addEventListener("change", () => {
    window.config.weather.showSunRays = sunRaysCheckboxEl.checked;       // updates the sunrays checkbox value in the config object
  });

  const waterBtn = document.getElementById("waterBtn");
  waterBtn.addEventListener("click", () => {
    if (typeof ui.onWater === "function") {                              // checks if the onWater callback has been set, and if so, executes it
      ui.onWater();
    }
  });
  
  // REGENERATION
  const regenerateBtn = document.getElementById("regenerateBtn");
  regenerateBtn.addEventListener("click", () => {
    if (typeof ui.onRegenerate === "function") {                         // checks if the onRegenerate callback exists and calls it when the button is clicked
      ui.onRegenerate(ui);
    }
  });

  // Initialize visual values ​​from configuration
  ui.updateControlsFromConfig();  // calls updateControlsFromConfig (see line 13) to ensure the UI is synchronized with the default config values when the application first loads
}

// Expose globally
window.ui = ui;                   // exposes the ui object globally by attaching it to the window object, making it accessible from other scripts (like bloomscape.js)
window.uiInit = uiInit;           // exposes the uiInit function globally so it can be called from the main script after the page has loaded
