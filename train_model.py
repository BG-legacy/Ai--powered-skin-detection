from Model import SkinCancerModel

if __name__ == "__main__":
    metadata_path = './HAM10000_metadata.csv'
    images_dir = './HAM10000_images_part_1'
    model = SkinCancerModel()
    model.train_with_ham10000(metadata_path, images_dir)
    print("Training complete!")
