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
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ composition }),
      });
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
      {/* Only show the form if there is no prediction */}
      {prediction === null && !loading && !error && (
        <ChemicalInput onPredict={handlePredict} />
      )}

      {/* Show loading, error, or prediction */}
      <div className="mt-8 flex justify-center">
        {loading && <p className="text-blue-600 text-lg">Predicting...</p>}
        {error && <p className="text-red-600 text-lg">{error}</p>}
        {prediction !== null && !loading && !error && (
          <div
            style={{
              maxWidth: 480,
              margin: '2rem auto',
              padding: '2.5rem 2rem',
              background: 'linear-gradient(135deg, #dbeafe 0%, #f1f5f9 100%)',
              borderRadius: '1.5rem',
              boxShadow: '0 4px 24px rgba(37,99,235,0.08)',
              textAlign: 'center'
            }}
          >
            <div style={{ fontSize: 56, color: '#16a34a', marginBottom: 12 }}>✔️</div>
            <h2 style={{ fontSize: '2.2rem', fontWeight: 700, color: '#1e293b', marginBottom: 16 }}>
              Predicted Ionic Conductivity
            </h2>
            <div style={{
              fontSize: '2.4rem',
              fontWeight: 700,
              color: '#2563eb',
              marginBottom: 10,
              letterSpacing: 1
            }}>
              {Number(prediction).toExponential(2).replace('e-', ' × 10⁻ ') +  '       S/cm'}
            </div>
            <button
              style={{
                padding: '0.75rem 2rem',
                fontSize: '1.1rem',
                fontWeight: 600,
                background: '#2563eb',
                color: '#fff',
                border: 'none',
                borderRadius: '0.5rem',
                cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(37,99,235,0.09)'
              }}
              onClick={() => setPrediction(null)}
            >
              Try Another
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
