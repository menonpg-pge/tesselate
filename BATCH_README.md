# SageMaker Batch Image Tesselation System

Complete solution for processing thousands of TIFF images in SageMaker with automatic slicing, metadata generation, and S3 integration.

## 🎯 **System Overview**

### **Input**: 12GB archive with nested TIFF images
### **Process**: Parallel tesselation into 8 pieces per image
### **Output**: Organized slices + metadata + comprehensive reports
### **Deploy**: SageMaker Jupyter Lab environment

## 📁 **File Structure**

```
tesselate/
├── batch_tesselate.py              # Main batch processing engine
├── enhanced_tesselate.py           # Core tesselation functions
├── SageMaker_Batch_Processing.ipynb # Ready-to-use Jupyter notebook
├── BATCH_README.md                 # This documentation
└── tesselate.py                    # Basic tesselation functions
```

## 🚀 **Quick Start Guide**

### **Step 1: Upload Files to SageMaker**
```bash
# Upload these files to your SageMaker Jupyter Lab:
batch_tesselate.py
enhanced_tesselate.py
SageMaker_Batch_Processing.ipynb
```

### **Step 2: Configure Paths**
Edit the notebook configuration:
```python
INPUT_DIR = '/home/ec2-user/SageMaker/your_12gb_dataset'
OUTPUT_DIR = '/home/ec2-user/SageMaker/processed_output'
```

### **Step 3: Run Processing**
Execute the notebook cells to:
1. Scan for TIFF images
2. Process in parallel batches  
3. Generate comprehensive reports
4. Create ZIP archive for download

## ⚙️ **Configuration Options**

### **BatchJob Parameters**

```python
job_config = BatchJob(
    input_dir=INPUT_DIR,        # Source directory
    output_dir=OUTPUT_DIR,      # Output directory
    num_pieces=8,               # Slices per image
    max_workers=4,              # Parallel workers
    min_image_size=1024,        # Skip images < 1024px
    skip_existing=True,         # Resume capability
    resume_mode=True,           # Load previous state
    log_level='INFO'            # Logging verbosity
)
```

### **Processing Logic**

#### **✅ PROCESSES:**
- `.tif` and `.tiff` files only
- Images ≥ 1024×1024 pixels
- Folders containing TIFF files

#### **❌ SKIPS:**
- JPG-only folders
- Images smaller than minimum size  
- Already processed images (if `resume_mode=True`)
- Non-image files

## 📊 **Output Structure**

### **Before Processing:**
```
12gb_dataset/
├── project_alpha/
│   ├── survey_001/
│   │   ├── aerial_photo_001.tif    ← Process this
│   │   ├── thumbnail_001.jpg       ← Skip this  
│   │   └── aerial_photo_002.tif    ← Process this
│   └── survey_002/
└── project_beta/
    └── crops/                      ← Skip (JPG-only folder)
        ├── crop_001.jpg            
        └── crop_002.jpg
```

### **After Processing:**
```
processed_output/
├── project_alpha/
│   ├── survey_001/
│   │   ├── aerial_photo_001/
│   │   │   ├── metadata.json          # Tesselation metadata
│   │   │   ├── batch_metadata.json    # Batch processing info
│   │   │   ├── slice_0_0_0_1024_768.png
│   │   │   ├── slice_1_1024_0_2048_768.png
│   │   │   └── ... (8 slices total)
│   │   └── aerial_photo_002/
│   │       └── ... (similar structure)
│   └── survey_002/
├── batch_state.json              # Resume state
├── batch_report.json             # Final summary
├── batch_processing.log          # Detailed logs
└── processed_output_results.zip  # Download package
```

## 🔧 **Advanced Features**

### **1. Smart Folder Detection**
- **Analyzes folder contents** before processing
- **Skips JPG-only folders** automatically  
- **Preserves directory structure** in output

### **2. Resume Capability**
- **Automatic state saving** every 10 images
- **Resume interrupted jobs** from exact stopping point
- **Skip already processed** images

### **3. Parallel Processing**
- **Multi-process execution** for speed
- **Configurable worker count** based on instance specs
- **Memory-efficient processing** for large TIFF files

### **4. Comprehensive Reporting**
```json
{
  "batch_summary": {
    "total_images_found": 1247,
    "successfully_processed": 1189,
    "skipped": 45,
    "errors": 13,
    "folders_skipped": 23,
    "total_processing_time": 14580.5,
    "average_time_per_image": 12.3,
    "images_per_hour": 293.4
  }
}
```

## 📈 **Performance Optimization**

### **SageMaker Instance Recommendations**

| Instance Type | vCPUs | RAM | Recommended Workers | Est. Speed |
|---------------|-------|-----|-------------------|------------|
| ml.t3.large   | 2     | 8GB | 2                 | ~50 img/hr |
| ml.m5.xlarge  | 4     | 16GB| 4                 | ~100 img/hr|
| ml.m5.2xlarge | 8     | 32GB| 6                 | ~180 img/hr|
| ml.m5.4xlarge | 16    | 64GB| 8                 | ~300 img/hr|

### **Memory Management**
- **Processes one image at a time** per worker
- **Automatic cleanup** of temporary data
- **Optimized for large TIFF files** (handles multi-GB images)

## 🛠️ **Command Line Usage**

### **Direct Python Execution**
```bash
python batch_tesselate.py \
  /path/to/input/dataset \
  /path/to/output/directory \
  --num-pieces 8 \
  --max-workers 4 \
  --min-size 1024 \
  --resume
```

### **Custom Processing Script**
```python
from batch_tesselate import BatchTesselator, BatchJob

# Configure job
config = BatchJob(
    input_dir="your_dataset_path",
    output_dir="output_path", 
    num_pieces=12,              # More pieces for larger images
    max_workers=6,              # More workers for faster processing
    min_image_size=2048         # Higher quality threshold
)

# Run processing
processor = BatchTesselator(config)
report = processor.run_batch()
```

## 🔍 **Verification & Quality Control**

### **Built-in Verification**
```python
from enhanced_tesselate import verify_images_identical

# Verify reconstruction quality
result = verify_images_identical(
    "original.tif", 
    "reconstructed.tif"
)
print(f"Images identical: {result['overall_identical']}")
```

### **Batch Verification Script**
```python
# Verify all processed images
def verify_batch_quality(output_dir):
    for metadata_file in Path(output_dir).rglob("metadata.json"):
        # Load metadata and verify reconstruction
        with open(metadata_file) as f:
            meta = json.load(f)
        
        original_path = meta["original_image"]["path"]
        # Reconstruct and verify...
```

## 🚨 **Error Handling**

### **Common Issues & Solutions**

#### **1. "No TIFF images found"**
- ✅ Check input directory path
- ✅ Verify TIFF file extensions (.tif/.tiff)
- ✅ Ensure folders aren't JPG-only

#### **2. "Permission denied" errors**
- ✅ Check SageMaker file permissions
- ✅ Ensure output directory is writable

#### **3. "Out of memory" errors**
- ✅ Reduce `max_workers` count
- ✅ Increase SageMaker instance size
- ✅ Check for extremely large TIFF files

#### **4. Processing stops/hangs**
- ✅ Check `batch_processing.log` for details
- ✅ Use resume mode to continue
- ✅ Reduce worker count if resource contention

## 📦 **S3 Integration Workflow**

### **Complete Workflow:**
```bash
# 1. Upload 12GB dataset to SageMaker
# 2. Run batch processing (creates ZIP)
# 3. Download ZIP from SageMaker 
# 4. Upload to S3

aws s3 cp processed_output_results.zip s3://your-bucket/processed-images/
```

### **Direct S3 Integration** (Optional Enhancement)
```python
import boto3

def upload_to_s3(local_dir, bucket, prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            s3_path = os.path.join(prefix, os.path.relpath(local_path, local_dir))
            s3.upload_file(local_path, bucket, s3_path)
```

## 🎯 **Expected Results**

### **For 12GB Dataset (est. 1000-2000 TIFF images):**
- **Processing time**: 6-12 hours (depending on instance)
- **Output size**: 15-25GB (due to PNG slices + metadata)
- **Success rate**: >95% (with detailed error reporting)
- **Resumable**: Can interrupt and continue anytime

### **Quality Guarantees:**
- ✅ **Lossless slicing** - Perfect reconstruction possible
- ✅ **Complete metadata** - Full traceability
- ✅ **Error isolation** - Single image failures don't stop batch
- ✅ **Comprehensive logging** - Full audit trail

## 🚀 **Ready to Deploy**

The system is **production-ready** with:
- ✅ **Robust error handling**
- ✅ **Resume capability** 
- ✅ **Comprehensive logging**
- ✅ **Performance optimization**
- ✅ **Quality verification**
- ✅ **SageMaker integration**

**Just upload the files to SageMaker and run the notebook!**
