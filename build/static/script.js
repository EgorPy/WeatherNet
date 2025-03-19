document.getElementById("delimiter").addEventListener("change", function() {
    document.getElementById("custom-delimiter").style.display = this.value === "custom" ? "block" : "none";
});

document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("file-input");
    if (!fileInput.files.length) return;

    let delimiter = document.getElementById("delimiter").value;
    if (delimiter === "custom") {
        delimiter = document.getElementById("custom-delimiter").value;
    }

    const ignoreHeader = document.getElementById("ignore-header");
    console.log(ignoreHeader.checked);

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("delimiter", delimiter);
    formData.append("ignore_header", ignoreHeader.checked);

    try {
        const response = await fetch("/upload/", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Ошибка загрузки");
        }

        document.getElementById("message").textContent = result.message || "Файл успешно загружен";
    } catch (error) {
        document.getElementById("message").textContent = `Ошибка: ${error.message}`;
    }
});
