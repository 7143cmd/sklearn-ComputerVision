import joblib
import os
from generate.generate import predict_color

def main():
    try:
        model = joblib.load('color_clasifff.pkl')
        
        target_folder = 'target_folder'
        files = os.listdir(target_folder)
        
        if not files:
            print("target folder is empty!")
            return
        
        first_file = sorted(files)[0]
        file_path = os.path.join(target_folder, first_file)
        
        result = predict_color(model, path=file_path)[0]
        print(f"Result: {result}")
        return result
        
    except Exception as e:
        print('Model not found, run generate.py manualy')

if __name__  == '__main__':
    main()