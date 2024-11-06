from Model import SkinCancerModel
import os
import pandas as pd
import logging

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    metadata_path = './HAM10000_metadata.csv'
    images_dir = './HAM10000_images_part_1'
    
    # Add path existence checks
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")
    
    # Read metadata file
    try:
        metadata_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata with {len(metadata_df)} entries")
        
        # Verify all images exist
        missing_images = []
        for image_id in metadata_df['image_id']:
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                missing_images.append(image_id)
        
        if missing_images:
            logger.warning(f"Missing {len(missing_images)} images: {missing_images[:5]}...")
        else:
            logger.info("All images found in directory")
        
        model = SkinCancerModel()
        model.train_with_ham10000(metadata_path, images_dir)
        print("Training complete!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
