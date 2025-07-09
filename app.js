import React, { useState } from 'react';
import ChemicalInput from './ChemicalInput';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (composition) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ composition }),
      });
      if (!response.ok) {
        throw new Error('Server error');
      }
      const data = await response.json();
      setPrediction(data.prediction);
    } catch (err) {
      setError('Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <ChemicalInput onPredict={handlePredict} />
      <div className="mt-8 flex justify-center">
        {loading && <p className="text-blue-600 text-lg">Predicting...</p>}
        {error && <p className="text-red-600 text-lg">{error}</p>}
        {prediction !== null && !loading && !error && (
          <div className="bg-white/90 rounded-xl shadow-md px-8 py-6 text-center">
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Predicted Ionic Conductivity
            </h2>
            <p className="text-3xl font-bold text-blue-700">{prediction}</p>
            <p className="text-gray-500 text-sm mt-2">
              (units depend on your model, e.g., log(S/cm))
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
