import os
import requests
from duckduckgo_search import DDGS
from urllib.parse import urlparse

def scrape_images(query, max_images, download_path="training-images"):
    """
    Searches for images using DuckDuckGo and downloads them.

    Args:
        query (str): The search term for the images.
        max_images (int): The maximum number of images to download.
        download_path (str): The directory to save the images to.
    """
    # Ensure the download directory exists
    os.makedirs(download_path, exist_ok=True)

    print(f"Searching for '{query}' and downloading up to {max_images} images...")
    
    downloaded_count = 0
    with DDGS() as ddgs:
        # Using DDGS.images() generator
        image_generator = ddgs.images(
            query,
            region="wt-wt",
            safesearch="off",
            size=None,
            color=None,
            type_image="photo",
            layout="Wide",
            license_image=None,
        )

        for result in image_generator:
            if downloaded_count >= max_images:
                break

            image_url = result.get("image")
            if not image_url:
                continue

            try:
                print(f"Downloading image {downloaded_count + 1}/{max_images} from {image_url}")
                response = requests.get(image_url, stream=True, timeout=15)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Create a simple filename
                file_extension = os.path.splitext(urlparse(image_url).path)[1]
                if not file_extension or len(file_extension) > 5:
                    # Guess extension from content type if possible
                    content_type = response.headers.get('content-type')
                    if content_type and 'jpeg' in content_type:
                        file_extension = '.jpg'
                    elif content_type and 'png' in content_type:
                        file_extension = '.png'
                    else:
                        file_extension = '.jpg' # default

                filename = f"image{downloaded_count + 1}{file_extension}"
                filepath = os.path.join(download_path, filename)

                # Save the image
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                
                print(f"Saved to {filepath}")
                downloaded_count += 1

            except requests.exceptions.RequestException as e:
                print(f"Could not download {image_url}. Error: {e}")
            except Exception as e:
                print(f"An error occurred with URL {image_url}: {e}")

    print(f"\nFinished. Downloaded {downloaded_count} images.")

if __name__ == "__main__":
    search_query = input("Enter the search query for images: ")
    num_images = int(input("How many images would you like to download? "))
    scrape_images(search_query, num_images)