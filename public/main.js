const btnSwitch = document.querySelector('#theme');
btnSwitch.addEventListener('click', () => {
    document.body.classList.toggle('dark');
    btnSwitch.classList.toggle('active');
})
<script>
  async function sendPrediction() {
    // Example input data - adjust keys to your model input columns
    const data = {
      feature1: 0.75,
      feature2: 0.23,
      feature3: 0.81,
      feature4: 0.12,
      feature5: 0.45
    };

    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await response.json();
    alert("Predicted score: " + result.prediction);
  }
</script>
