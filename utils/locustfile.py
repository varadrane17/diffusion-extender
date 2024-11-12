from locust import HttpUser, task

class APIUser(HttpUser):

    def on_start(self):
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': "Basic dXNlcjphcHB5aGlnaEAzMjE="
        }  

        # self.json_data = "Parameters in terms of JSON" # Put you parameters to your API client in json format
        self.json_data = {
            "order_id": "rishabh",
            "image_url": "https://ai-image-editor-wasabi-bucket.apyhi.com/uncrop_assets/original_images/image%2010.webp",
            "width": 1024,
            "height": 1024,
            "overlap_width": 100,
            "prompt_input": "",
            "resize_option": "idk",
            "num_inference_steps": 10,
            "margin_x": 10,
            "margin_y": 10,
        }

    @task
    def generate_image(self):
        with self.client.post('/generate-image', # API Endpoint
                              headers=self.headers,
                              json=self.json_data,
                              catch_response=True) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

def run_load_test(users, duration):
    from subprocess import call

    # Construct the command
    cmd = f"locust -f {__file__} --headless -u {users} -r {users} --run-time {duration}s --host=http://localhost:8014" # Replace host with your API host
    
    # Run the command
    call(cmd, shell=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load test for API")
    parser.add_argument("--users", type=int, default=10, help="Number of users to simulate")
    parser.add_argument("--duration", type=int, default=300, help="Duration of the test in seconds")
    
    args = parser.parse_args()
    
    print(f"Starting load test with {args.users} users for {args.duration} seconds...")
    run_load_test(args.users, args.duration)