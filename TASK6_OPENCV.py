import numpy as np
import cv2

cap = cv2.VideoCapture("volleyball_match.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

#net position
NET_Y = 322

# court polygon vertices
court_vertices = np.array([(122, 709), (1200, 709), (984, 200), (335, 200)])

def is_inside_court(x, y):
    #to check if the point is inside the court or not
    return cv2.pointPolygonTest(court_vertices, (x, y), False) >= 0

def detect_ball(frame):
    
    image_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    lower_yellow = np.array([75, 133, 0], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 129], dtype=np.uint8)
    yellow_mask = cv2.inRange(image_rgb, lower_yellow, upper_yellow)

    image_yuv = cv2.cvtColor(frame,cv2.COLOR_BGR2YUV)
    yuv_lower = np.array([51, 65, 139])
    yuv_upper = np.array([227, 102, 182])
    yuv_mask = cv2.inRange(image_yuv, yuv_lower, yuv_upper)
    result = cv2.bitwise_or(yellow_mask,yuv_mask)

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01* cv2.arcLength(cnt, True), True)
        if len(approx)>12:
            ((x, y), radius) = cv2.minEnclosingCircle(approx)
            if radius < 10 and radius > 1 and y<210:
                cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 2)
                cv2.putText(frame, 'volleyball', (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),2)
                
def detect_teams(frame):
    fg_mask = bg_subtractor.apply(frame)
    
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    team1 = []
    team2 = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  
            x, y, w, h = cv2.boundingRect(cnt)
            center_x, center_y = x + w // 2, y + h // 2 

            if is_inside_court(center_x, center_y):  
                if center_y < NET_Y:
                    team1.append((x, y, w, h))  
                else:
                    team2.append((x, y, w, h))  

    return team1, team2

# processing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  
    
    ball = detect_ball(frame)    
    
    team1, team2 = detect_teams(frame)
    
    for (x, y, w, h) in team1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    for (x, y, w, h) in team2:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.putText(frame, f"Team 1: {len(team1)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Team 2: {len(team2)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Volleyball Tracking", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
