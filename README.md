
</head>
<body>
  <div class="section">
    <h1>Building Damage Assessment Application</h1>
    <p>This project is a web-based application designed to assess damage to buildings by comparing "before" and "after" images. It uses a combination of YOLOv8 object detection and K-means clustering to identify and analyze building structures, detecting changes that indicate potential damage. The backend is built with <a href="https://fastapi.tiangolo.com">FastAPI</a>, and the frontend is developed using <a href="https://reactjs.org">React</a> with <a href="https://tailwindcss.com">Tailwind CSS</a> for styling.</p>
  </div>

  <div class="section">
    <h2>Features</h2>
    <ul>
      <li><strong>Image Upload</strong>: Upload "before" and "after" images for damage assessment.</li>
      <li><strong>Building Detection</strong>: Utilizes YOLOv8 for object detection and K-means clustering to identify buildings.</li>
      <li><strong>Damage Assessment</strong>: Compares pixel differences in building clusters to estimate damage.</li>
      <li><strong>Visualization</strong>: Displays clustered images, images with bounding boxes, difference images, and detailed cluster information.</li>
      <li><strong>API Endpoints</strong>: Provides RESTful endpoints for fetching, uploading, and deleting assessment data.</li>
      <li><strong>Responsive UI</strong>: Built with React and Tailwind CSS for a modern, user-friendly interface.</li>
      <li><strong>Error Handling</strong>: Robust error messages for invalid inputs or server issues.</li>
    </ul>
  </div>

  <div class="section">
    <h2>Model</h2>
    <p>The application uses a pre-trained <strong>YOLOv8</strong> model for building detection, stored at <code>backend/dataset/runs/detect/train2/weights/best.pt</code>. The model is configured to detect buildings with a confidence threshold of 0.5. K-means clustering (with k=8) is applied to complement YOLOv8 by identifying additional building regions based on pixel color, particularly focusing on white building clusters (RGB values between 150 and 255).</p>
    <h3>Detection Process</h3>
    <ol>
      <li><strong>YOLOv8 Detection</strong>: Identifies buildings with bounding boxes.</li>
      <li><strong>K-means Clustering</strong>: Segments remaining image regions to detect additional buildings not identified by YOLOv8.</li>
      <li><strong>Non-Maximum Suppression</strong>: Merges overlapping boxes (IoU > 0.3) to avoid duplicates.</li>
      <li><strong>Damage Analysis</strong>: Compares white building cluster pixels between "_are" and "after" images to quantify changes.</li>
    </ol>
  </div>

  <div class="section">
    <h2>API Endpoints</h2>
    <ul>
      <li>
        <strong>GET /fruits</strong>
        <p><strong>Description</strong>: Retrieves stored clustering and damage assessment results.</p>
        <p><strong>Response</strong>: JSON object containing a list of assessment results, including clusters, clustered images, difference images, and damage summaries.</p>
      </li>
      <li>
        <strong>POST /images</strong>
        <p><strong>Description</strong>: Uploads "before" and "after" images for processing.</p>
        <p><strong>Parameters</strong>:</p>
        <ul>
          <li><code>beforeImage</code>: Image file (JPEG/PNG).</li>
          <li><code>afterImage</code>: Image file (JPEG/PNG).</li>
          <li><code>skip_ssim</code>: Boolean (optional, default: false) to skip SSIM validation.</li>
        </ul>
        <p><strong>Response</strong>: JSON object with clustering results, base64-encoded images, and damage summary.</p>
        <p><strong>Validation</strong>: Ensures images are of the same scene (SSIM > 0.2) unless <code>skip_ssim</code> is true.</p>
      </li>
      <li>
        <strong>DELETE /fruits</strong>
        <p><strong>Description</strong>: Clears all stored assessment results.</p>
        <p><strong>Response</strong>: Confirmation message.</p>
      </li>
    </ul>
  </div>

  <div class="section">
    <h2>Requirements</h2>
    <h3>Backend</h3>
    <ul>
      <li><strong>Python</strong>: 3.8 or higher</li>
      <li><strong>Dependencies</strong> (listed in <code>requirements.txt</code>):</li>
      <ul>
        <li><code>fastapi</code></li>
        <li><code>uvicorn</code></li>
        <li><code>opencv-python</code></li>
        <li><code>numpy</code></li>
        <li><code>pydantic</code></li>
        <li><code>ultralytics</code></li>
        <li><code>scikit-image</code></li>
        <li><code>python-multipart</code></li>
        <li><code>python-logging</code></li>
      </ul>
    </ul>
    <h3>Frontend</h3>
    <ul>
      <li><strong>Node.js</strong>: 16 or higher</li>
      <li><strong>Dependencies</strong> (listed in <code>package.json</code>):</li>
      <ul>
        <li><code>react</code></li>
        <li><code>axios</code></li>
        <li><code>tailwindcss</code></li>
        <li><code>vite</code> (or other build tool)</li>
        <li>Other dependencies as specified in <code>package.json</code></li>
      </ul>
    </ul>
  </div>

  <div class="section">
    <h2>Installation</h2>
    <h3>Backend</h3>
    <ol>
      <li>Clone the repository:
        <pre><code>git clone &lt;repository-url&gt;
cd backend</code></pre>
      </li>
      <li>Create a virtual environment and activate it:
        <pre><code>python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate</code></pre>
      </li>
      <li>Install dependencies:
        <pre><code>pip install -r requirements.txt</code></pre>
      </li>
      <li>Update the YOLO model path in the backend code:
        <pre><code>model = YOLO('path/to/your/best.pt')</code></pre>
      </li>
      <li>Run the FastAPI server:
        <pre><code>uvicorn main:app --host 0.0.0.0 --port 8000</code></pre>
      </li>
    </ol>
    <h3>Frontend</h3>
    <ol>
      <li>Navigate to the frontend directory:
        <pre><code>cd frontend</code></pre>
      </li>
      <li>Install dependencies:
        <pre><code>npm install</code></pre>
      </li>
      <li>Configure the API base URL in <code>src/api.js</code>:
        <pre><code>import axios from 'axios';
export default axios.create({
  baseURL: 'http://localhost:8000',
});</code></pre>
      </li>
      <li>Run the development server:
        <pre><code>npm run dev</code></pre>
      </li>
      <li>Access the application at <code>http://localhost:5173</code>.</li>
    </ol>
  </div>

  <div class="section">
    <h2>Usage</h2>
    <ol>
      <li>Start the backend server:
        <pre><code>uvicorn main:app --host 0.0.0.0 --port 8000</code></pre>
      </li>
      <li>Start the frontend server:
        <pre><code>npm run dev</code></pre>
      </li>
      <li>Open your browser to <code><a href="http://localhost:5173">http://localhost:5173</a></code>.</li>
      <li>Upload "before" and "after" images via the provided form.</li>
      <li>View the clustering results, bounding box images, difference image, and damage summary.</li>
      <li>Use the "Delete" button to clear stored results.</li>
    </ol>
  </div>

  <div class="section">
    <h2>Directory Structure</h2>
    <pre><code>project/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── dataset/runs/detect/train2/weights/best.pt
├── frontend/
│   ├── src/
│   │   ├── api.js
│   │   ├── Fruits.jsx
│   │   ├── AddFruitForm.jsx
│   │   └── App.jsx
│   ├── package.json
│   └── tailwind.config.js
└── README.md
</code></pre>
  </div>

  <div class="section">
    <h2>Notes</h2>
    <ul>
      <li><strong>CORS Configuration</strong>: The backend allows CORS requests from <code>http://localhost:5173</code>. Update the <code>origins</code> list in <code>main.py</code> if using a different frontend URL.</li>
      <li><strong>SSIM Validation</strong>: The application checks for scene similarity (SSIM > 0.2) to ensure "before" and "after" images depict the same scene. This can be bypassed by setting <code>skip_ssim=true</code> in the POST request.</li>
      <li><strong>Model Path</strong>: Ensure the YOLO model path is correctly set to avoid file loading errors.</li>
      <li><strong>Performance</strong>: Image processing can be resource-intensive. Ensure sufficient CPU/GPU resources for large images or high-resolution processing.</li>
    </ul>
  </div>

  <div class="section">
    <h2>Troubleshooting</h2>
    <ul>
      <li><strong>Backend Connection Error</strong>: Verify the FastAPI server is running at <code>http://localhost:8000</code>.</li>
      <li><strong>Image Upload Failure</strong>: Ensure images are valid JPEG/PNG files and meet the SSIM threshold (or use <code>skip_ssim</code>).</li>
      <li><strong>Model Loading Error</strong>: Check the YOLO model path and file permissions.</li>
      <li><strong>CORS Issues</strong>: Ensure the frontend URL is included in the <code>origins</code> list in <code>main.py</code>.</li>
    </ul>
  </div>

  <div class="section">
    <h2>Future Improvements</h2>
    <ul>
      <li>Add authentication for secure API access.</li>
      <li>Implement batch processing for multiple image pairs.</li>
      <li>Enhance damage assessment with more detailed metrics (e.g., percentage of damage).</li>
      <li>Add support for different image formats or resolutions.</li>
      <li>Integrate real-time progress updates during image processing.</li>
    </ul>
  </div>

  <div class="section">
    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for details.</p>
  </div>
</body>
</html>
