
# coding: utf-8

# In[1]:

import numpy as np
import cv2


# In[2]:


dim = 100

def resize_image(img, size ):
    
    img = cv2.resize(img, (size, size))
    print("Image successfully resized!")
    return img


# In[75]:

def remove_background(img):
#     making background black

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
#     for accuracy making two masks to detect skin 
    
#     mask 1
    lower_boundary = np.array([0, 40, 30], dtype="uint8")
    upper_boundary = np.array([43, 255, 254], dtype="uint8")
    skin_mask = cv2.inRange(img, lower_boundary, upper_boundary)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

#     mask 2
    lower_boundary = np.array([170, 80, 30], dtype="uint8")
    upper_boundary = np.array([180, 255, 250], dtype="uint8")
    skin_mask2 = cv2.inRange(img, lower_boundary, upper_boundary)
    
#     combined mask
    skin_mask = cv2.addWeighted(skin_mask, 0.5, skin_mask2, 0.5, 0.0)
    
    skin_mask = cv2.medianBlur(skin_mask, 5)
    
    
    img_skin = cv2.bitwise_and(img, img, mask=skin_mask)
    
    img = cv2.addWeighted(img, 1.5, img_skin, -0.5, 0)
    
    img_skin = cv2.bitwise_and(img, img, mask=skin_mask)
    
    print("Success!")
    return img_skin
    
    


# In[19]:

def detecting_hand(img):
    # makes hand white
    
    h, w = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # setting skin white
    thresh = 1
    for i in range(h):
        for j in range(w):
            
            if img[i][j] > thresh:
                img[i][j] = 255
            else:
                img[i][j] = 0

    print("Success!")
    return img
    
    


# In[5]:

def remove_arm(img):
    
    h, w = img.shape[:2]
    img = img[:h - 15, :]
    print("Success!")
    
    return (img)


# In[21]:

def draw_contours(img):
  
  
    _,contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #finding largest contour
    
    if len(contours) <= 0:
        print("error!")
        
    largest_contour_index = 0

    contour = 1
    while contour < len(contours):
        
        if cv2.contourArea(contours[contour]) > cv2.contourArea(contours[largest_contour_index]):
            largest_contour_index = contour
            
        contour += 1

        
    cv2.drawContours(img, contours, largest_contour_index, (255, 255, 255), thickness=-1)

    contour_dimensions = cv2.boundingRect(contours[largest_contour_index])
    
    print("Done!")
    return (img, contour_dimensions)


# In[84]:

def apply_preprocessing(img):
   
    img = resize_image(img, dim)
    img = remove_background(img)
    img = detecting_hand(img)
    img = remove_arm(img)
    img, contour_dimensions = draw_contours(img)
    img = resize_image(img, dim)
    return img


# In[8]:

# paths = 'C:/Users/Pakhi/Hackerrank - predict annual returns/ASL/FinalASL/asl_dataset/0'


# In[77]:

# img = cv2.imread('Capture.JPG')
# img = cv2.resize(img,(50, 50))   


# In[82]:

# cv2.imshow("oe",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[81]:

# img = apply_preprocessing(img)


# In[ ]:



