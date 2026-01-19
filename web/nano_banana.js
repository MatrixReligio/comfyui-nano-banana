// nano_banana.js - Frontend extension for Nano Banana nodes
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "Comfy.NanoBanana",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to both Text to Image and Image to Image nodes
        if (nodeData.name === "NanoBananaTextToImage" || nodeData.name === "NanoBananaImageToImage") {
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                // Store model widget reference
                this.modelWidget = this.widgets.find(w => w.name === "model_name");

                // Add Verify API Key button
                const verifyBtn = this.addWidget("button", "Verify API Key", null, () => {
                    this.verifyApiKey();
                });
                verifyBtn.serialize = false;

                // Add Update Models button
                const updateModelsBtn = this.addWidget("button", "Update Models", null, () => {
                    this.updateModels(true);  // true = show alert
                });
                updateModelsBtn.serialize = false;

                // Auto-update models on node creation (delayed, silent)
                setTimeout(() => {
                    this.updateModels(false);  // false = silent
                }, 2000);
            };

            // Verify API Key function
            nodeType.prototype.verifyApiKey = function() {
                console.log("Nano Banana: Verifying API key...");

                fetch("/nano_banana/check_api_key", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({})
                })
                .then(response => response.json())
                .then(data => {
                    console.log("Nano Banana: API key check result:", data);
                    if (data.status === "success") {
                        alert("API Key Valid!\n\n" + data.message);
                        // Also update models after successful verification
                        this.updateModels(false);
                    } else {
                        alert("API Key Error!\n\n" + data.message);
                    }
                })
                .catch(error => {
                    console.error("Nano Banana: Error checking API key:", error);
                    alert("Error checking API key:\n\n" + error);
                });
            };

            // Update Models function
            // showAlert: if true, show alert with results
            nodeType.prototype.updateModels = function(showAlert = true) {
                console.log("Nano Banana: Fetching models...");

                fetch("/nano_banana/get_models", {
                    method: "GET",
                    headers: { "Content-Type": "application/json" }
                })
                .then(response => response.json())
                .then(data => {
                    console.log("Nano Banana: Models result:", data);
                    if (data.models && data.models.length > 0) {
                        // Update model widget options
                        if (this.modelWidget) {
                            const currentModel = this.modelWidget.value;
                            this.modelWidget.options.values = data.models;

                            // Keep current model if it exists in new list
                            if (data.models.includes(currentModel)) {
                                this.modelWidget.value = currentModel;
                            } else {
                                this.modelWidget.value = data.models[0];
                            }

                            console.log("Nano Banana: Updated models list:", data.models);
                            this.setDirtyCanvas(true, true);
                        }

                        // Show alert with results only if requested
                        if (showAlert) {
                            alert("Models Updated!\n\nFound " + data.models.length + " models:\n\n" + data.models.slice(0, 5).join("\n") + (data.models.length > 5 ? "\n..." : ""));
                        }
                    } else {
                        if (showAlert) {
                            alert("No models found.\n\n" + (data.message || ""));
                        }
                    }
                })
                .catch(error => {
                    console.error("Nano Banana: Error fetching models:", error);
                    if (showAlert) {
                        alert("Error fetching models:\n\n" + error);
                    }
                });
            };
        }
    }
});

console.log("Nano Banana extension loaded");
