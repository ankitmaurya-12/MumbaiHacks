from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
import base64
from groq import Groq
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Initialize the Groq client with the API key
API_KEY = "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"
client = Groq(api_key=API_KEY)

class GroqResponse(BaseModel):
    message: str

@app.post("/analyze-medical-image", response_model=GroqResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image file data
        image_data = await file.read()
        
        # Check file size
        if len(image_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image exceeds maximum size")
        
        # Encode the image to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        try:
            # Create the Groq API request
            completion = client.chat.completions.create(
                model="llama2-70b-4096",  # Using LLaMA 2 model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical image analysis assistant. Analyze the provided medical image carefully and provide detailed observations."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please analyze this medical image in detail. Include:\n1. Type of scan/image\n2. Visible anatomical structures\n3. Any abnormalities or concerning findings\n4. Potential diagnoses\n5. Recommendations for further testing if needed"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.7,  # Reduced temperature for more focused responses
                max_tokens=2048,  # Increased max tokens for detailed analysis
                top_p=0.95,
                stream=False
            )
            
            # Extract the response
            if completion and completion.choices:
                message_content = completion.choices[0].message.content
                print("Analysis received:", message_content)  # Debug print
                return {"message": message_content}
            else:
                raise HTTPException(status_code=500, detail="No response received from analysis model")
                
        except Exception as groq_error:
            print(f"Groq API error: {str(groq_error)}")
            raise HTTPException(status_code=500, detail=f"Analysis service error: {str(groq_error)}")
    
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        print(f"General error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)