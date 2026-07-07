import os
import uuid
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tasks import train_model_task, celery_app
from flask import send_from_directory


app = Flask(__name__)

# 1. Enable CORS: Required because Next.js runs on port 3000 and Flask on port 5000
CORS(app) 

# This matches the volume mounted in our docker-compose.yml
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/app/shared_data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/train', methods=['POST'])
def train_endpoint():
    """Receives the CSV, saves it to disk, and triggers the Celery worker."""
    # ... file validation ...
    
    # file = request.files['file']
    # # .get() defaults to None if the key is missing or empty
    # target_column = request.form.get('target_column', '').strip() 
    
    # Remove the `if not target_column:` error block entirely.
    # Pass the variable as-is to Celery.
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    target_column = request.form.get('target_column', '').strip()  # Optional; can be empty for auto-clustering
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    # if not target_column:
    #     target_column = None  # Allow None for auto-clustering

    # Generate a unique job ID to track this specific training run
    job_id = str(uuid.uuid4())
    
    # Save file to the shared Docker volume safely
    safe_filename = f"{job_id}_{secure_filename(file.filename)}"
    file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(file_path)
    
    # 2. Dispatch the background task!
    # We pass the job_id into apply_async so Celery uses our UUID to track it
    task = train_model_task.apply_async(
        args=[file_path, target_column, job_id], 
        task_id=job_id
    )
    
    return jsonify({'job_id': task.id, 'status': 'QUEUED'}), 202


@app.route('/api/status/<job_id>', methods=['GET'])
def status_endpoint(job_id):
    """Next.js polls this endpoint every 3 seconds to check on the worker."""
    task = train_model_task.AsyncResult(job_id)
    
    if task.state == 'PENDING':
        return jsonify({'status': 'PENDING'})
    elif task.state not in ['FAILURE', 'SUCCESS']:
        # This handles the custom 'TRAINING' state we set in tasks.py
        return jsonify({'status': 'TRAINING'})
    elif task.state == 'SUCCESS':
        # task.info contains the dictionary we returned at the end of tasks.py
        return jsonify(task.info)
    else:
        # Task crashed (e.g., out of memory, invalid CSV data)
        return jsonify({'status': 'FAILED', 'error': str(task.info)}), 500


@app.route('/api/download/<job_id>', methods=['GET'])
def download_endpoint(job_id):
    """Sends the saved .joblib model back to the user's browser."""
    model_filename = f"model_{job_id}.joblib"
    file_path = os.path.join(UPLOAD_FOLDER, model_filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
        
    return jsonify({'error': 'Model not found'}), 404



@app.route('/api/report/<job_id>', methods=['GET'])
def get_report(job_id):
    """Serves the interactive profiling report or triggers a download."""
    filename = f"report_{job_id}.html"
    
    # Check if download query parameter is passed (?download=true)
    as_attachment = request.args.get('download', 'false').lower() == 'true'
    
    try:
        return send_from_directory(
            '/app/shared_data', 
            filename, 
            as_attachment=as_attachment,
            download_name=f"data_profile_{job_id}.html"
        )
    except FileNotFoundError:
        return {"error": "Report not found"}, 404


@app.route('/api/report/powerbi/<job_id>', methods=['GET'])
def get_powerbi_report(job_id):
    """Serves the raw JSON profiling data for PowerBI ingestion."""
    filename = f"report_{job_id}.json"
    
    try:
        return send_from_directory(
            '/app/shared_data', 
            filename, 
            as_attachment=True,
            download_name=f"powerbi_data_profile_{job_id}.json"
        )
    except FileNotFoundError:
        return {"error": "PowerBI JSON report not found"}, 404

if __name__ == '__main__':
    # Run on 0.0.0.0 so the Docker container exposes it to the network
    app.run(debug=True, host='0.0.0.0')