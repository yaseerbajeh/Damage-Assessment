import React, { useEffect, useState } from 'react';
import api from '../api.js';
import AddFruitForm from './AddFruitForm';

const Fruits = () => {
  const [fruits, setFruits] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchFruits = async () => {
    try {
      setLoading(true);
      const response = await api.get('/fruits');
      if (!response.data || !response.data.fruits) throw new Error('Invalid response structure');
      setFruits(response.data.fruits);
      setError(null);
    } catch (error) {
      console.error('Error fetching clustering data:', error);
      setError(
        error.response
          ? `Failed to fetch clustering data: Server error (${error.response.status})`
          : error.request
          ? 'Failed to fetch clustering data: Unable to connect to the backend. Please ensure it is running at http://localhost:8000.'
          : `Failed to fetch clustering data: ${error.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  const addFruit = async (beforeImageFile, afterImageFile) => {
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('beforeImage', beforeImageFile);
      formData.append('afterImage', afterImageFile);

      const response = await api.post('/images', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log('Upload response:', response.data);
      await new Promise((resolve) => setTimeout(resolve, 500));
      await fetchFruits();
      setError(null);
    } catch (error) {
      console.error('Error uploading images:', error.response || error);
      setError(
        error.response
          ? error.response.data.detail.includes('SSIM')
            ? `Failed to upload images: ${error.response.data.detail}. Try images of the same scene or contact support for assistance.`
            : `Failed to upload images: ${error.response.data.detail} (${error.response.status})`
          : error.request
          ? 'Failed to upload images: Unable to connect to the backend. Please ensure it is running at http://localhost:8000.'
          : `Failed to upload images: ${error.message}`
      );
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const deleteFruit = async (index) => {
    try {
      setLoading(true);
      await api.delete('/fruits');
      setFruits([]);
      setError(null);
    } catch (error) {
      console.error('Error deleting clustering data:', error);
      setError(
        error.response
          ? `Failed to delete clustering data: Server error (${error.response.status})`
          : error.request
          ? 'Failed to delete clustering data: Unable to connect to the backend.'
          : `Failed to delete clustering data: ${error.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFruits();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-extrabold text-gray-900 mb-8 tracking-wide">Automated Damage Assessment</h1>
        {error && (
          <div
            className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-lg mb-6 shadow-md transition-all duration-300"
            role="alert"
          >
            {error}
          </div>
        )}
        {loading && (
          <div className="bg-blue-50 border border-blue-200 text-blue-700 p-4 rounded-lg mb-6 shadow-md flex items-center animate-pulse">
            <svg
              className="w-5 h-5 mr-2 text-blue-500 animate-spin"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Processing images or deleting data, please wait...
          </div>
        )}
        {fruits.length === 0 && !error && !loading && (
          <div className="bg-gray-100 border border-gray-200 text-gray-600 p-4 rounded-lg mb-6 shadow-md">
            No assessment data available. Upload before and after images to analyze damage.
          </div>
        )}
        <div className="space-y-8">
          {fruits.map((fruit, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow duration-300 border border-gray-100"
            >
              <div className="flex justify-between items-start mb-6">
                <h2 className="text-2xl font-semibold text-gray-800">Assessment {index + 1}</h2>
                <button
                  onClick={() => deleteFruit(index)}
                  className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors duration-200 shadow-md hover:shadow-lg disabled:bg-gray-400 disabled:cursor-not-allowed"
                  disabled={loading}
                >
                  Delete
                </button>
              </div>
              {fruit.damage_summary && (
                <div className="bg-blue-50 border border-blue-100 text-blue-700 p-4 rounded-lg mb-6">
                  <h3 className="text-lg font-medium mb-2">Damage Summary</h3>
                  <p className="text-base whitespace-pre-line">{fruit.damage_summary.details}</p>
                </div>
              )}
              <div className="grid grid-cols-2 gap-6 mb-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">Before Clustered</h3>
                  {fruit.before_clustered_image ? (
                    <img
                      src={`data:image/jpeg;base64,${fruit.before_clustered_image}`}
                      alt="Before Clustered"
                      className="w-full h-64 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                    />
                  ) : (
                    <div className="bg-gray-100 text-gray-500 p-4 rounded-lg">No image available</div>
                  )}
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">After Clustered</h3>
                  {fruit.after_clustered_image ? (
                    <img
                      src={`data:image/jpeg;base64,${fruit.after_clustered_image}`}
                      alt="After Clustered"
                      className="w-full h-64 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                    />
                  ) : (
                    <div className="bg-gray-100 text-gray-500 p-4 rounded-lg">No image available</div>
                  )}
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">Before Detected</h3>
                  {fruit.before_detected_image ? (
                    <img
                      src={`data:image/jpeg;base64,${fruit.before_detected_image}`}
                      alt="Before with Detections"
                      className="w-full h-64 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                      onError={() => setError('Failed to load before image with detections')}
                    />
                  ) : (
                    <div className="bg-gray-100 text-gray-500 p-4 rounded-lg">No image available</div>
                  )}
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">After Detected</h3>
                  {fruit.after_detected_image ? (
                    <img
                      src={`data:image/jpeg;base64,${fruit.after_detected_image}`}
                      alt="After with Detections"
                      className="w-full h-64 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                      onError={() => setError('Failed to load after image with detections')}
                    />
                  ) : (
                    <div className="bg-gray-100 text-gray-500 p-4 rounded-lg">No image available</div>
                  )}
                </div>
              </div>
              {fruit.diff_image && (
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">Damaged Areas</h3>
                  <div className="bg-red-50 border border-red-200 p-4 rounded-lg shadow-md">
                    <img
                      src={`data:image/jpeg;base64,${fruit.diff_image}`}
                      alt="Damaged Areas"
                      className="w-full h-64 object-cover rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300"
                      onError={() => setError('Failed to load difference image')}
                    />
                  </div>
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">Before Image Clusters</h3>
                  {fruit.before_clusters && fruit.before_clusters.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse bg-white rounded-lg shadow-md">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="border p-2 text-left">Centroid</th>
                            <th className="border p-2 text-left">Pixels</th>
                            <th className="border p-2 text-left">RGB Values</th>
                            <th className="border p-2 text-left">Mean R</th>
                            <th className="border p-2 text-left">Mean G</th>
                            <th className="border p-2 text-left">Mean B</th>
                          </tr>
                        </thead>
                        <tbody>
                          {fruit.before_clusters.map((cluster) => (
                            <tr key={cluster.centroid_id} className="hover:bg-gray-50">
                              <td className="border p-2">{cluster.centroid_id}</td>
                              <td className="border p-2">{cluster.num_pixels}</td>
                              <td className="border p-2">
                                {cluster.centroid_rgb && cluster.centroid_rgb.length === 3
                                  ? `[${cluster.centroid_rgb[0].toFixed(2)}, ${cluster.centroid_rgb[1].toFixed(2)}, ${cluster.centroid_rgb[2].toFixed(2)}]`
                                  : 'N/A'}
                              </td>
                              <td className="border p-2">{cluster.mean_red.toFixed(2)}</td>
                              <td className="border p-2">{cluster.mean_green.toFixed(2)}</td>
                              <td className="border p-2">{cluster.mean_blue.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-gray-700">After Image Clusters</h3>
                  {fruit.after_clusters && fruit.after_clusters.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm border-collapse bg-white rounded-lg shadow-md">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="border p-2 text-left">Centroid</th>
                            <th className="border p-2 text-left">Pixels</th>
                            <th className="border p-2 text-left">RGB Values</th>
                            <th className="border p-2 text-left">Mean R</th>
                            <th className="border p-2 text-left">Mean G</th>
                            <th className="border p-2 text-left">Mean B</th>
                          </tr>
                        </thead>
                        <tbody>
                          {fruit.after_clusters.map((cluster) => (
                            <tr key={cluster.centroid_id} className="hover:bg-gray-50">
                              <td className="border p-2">{cluster.centroid_id}</td>
                              <td className="border p-2">{cluster.num_pixels}</td>
                              <td className="border p-2">
                                {cluster.centroid_rgb && cluster.centroid_rgb.length === 3
                                  ? `[${cluster.centroid_rgb[0].toFixed(2)}, ${cluster.centroid_rgb[1].toFixed(2)}, ${cluster.centroid_rgb[2].toFixed(2)}]`
                                  : 'N/A'}
                              </td>
                              <td className="border p-2">{cluster.mean_red.toFixed(2)}</td>
                              <td className="border p-2">{cluster.mean_green.toFixed(2)}</td>
                              <td className="border p-2">{cluster.mean_blue.toFixed(2)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        <AddFruitForm addFruit={addFruit} />
      </div>
    </div>
  );
};

export default Fruits;
