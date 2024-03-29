
#include <Arduino.h>
#include <Adafruit_MCP3008.h>
#include <Adafruit_MPU6050.h>
#include <Encoder.h>



/* WIFI BEGIN
 *  This sketch sends random data over UDP on a ESP32 device
 */
#include <WiFi.h>
#include <WiFiUdp.h>

// WiFi network name and password:
const char * networkName = "408ITerps";
const char * networkPswd = "goterps2022";

// Home Wifi name and password
// const char * networkName = "Fios-rKrV7";
// const char * networkPswd = "taste24roe67acre";

//IP address to send UDP data to:
// either use the ip address of the server or 
// a network broadcast address
const char * udpAddress = "192.168.2.108"; //NEED TO CHECK
const int mouse1_udpPort = 3334; //3334; for mouse2
//const int mouse1_udpPort = 3334;

boolean connected = false;

//The udp library class
WiFiUDP udp;


//wifi event handler
void WiFiEvent(WiFiEvent_t event){
    switch(event) {
      case ARDUINO_EVENT_WIFI_STA_GOT_IP:
          //When connected set 
          Serial.print("WiFi connected! IP address: ");
          Serial.println(WiFi.localIP());  
          //initializes the UDP state
          //This initializes the transfer buffer
          udp.begin(WiFi.localIP(),mouse1_udpPort);
          connected = true;
          break;
      case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
          Serial.println("WiFi lost connection");
          connected = false;
          break;
      default: break;
    }
}

void connectToWiFi(const char * ssid, const char * pwd){
  Serial.println("Connecting to WiFi network: " + String(ssid));

  // delete old config
  WiFi.disconnect(true);
  //register event handler
  WiFi.onEvent(WiFiEvent);
  
  //Initiate connection
  WiFi.begin(ssid, pwd);

  Serial.println("Waiting for WIFI connection...");
}

// IMU (rotation rate and acceleration)
Adafruit_MPU6050 mpu;
Adafruit_MCP3008 adc1;
Adafruit_MCP3008 adc2;

// Buzzer pin which we will use for indicating IMU initialization failure
const unsigned int BUZZ = 26;
const unsigned int BUZZ_CHANNEL = 0;

// Need these pins to turn off light bar ADC chips
const unsigned int ADC_1_CS = 2;
const unsigned int ADC_2_CS = 17;

// Battery voltage measurement constants
const unsigned int VCC_SENSE = 27;
const float ADC_COUNTS_TO_VOLTS = (2.4 + 1.0) / 1.0 * 3.3 / 4095.0;

// Motor encoder pins
const unsigned int M1_ENC_A = 39;
const unsigned int M1_ENC_B = 38;
const unsigned int M2_ENC_A = 37;
const unsigned int M2_ENC_B = 36;

// Motor power pins
const unsigned int M1_IN_1 = 13;
const unsigned int M1_IN_2 = 12;
const unsigned int M2_IN_1 = 25;
const unsigned int M2_IN_2 = 14;

// Motor PWM channels
const unsigned int M1_IN_1_CHANNEL = 8;
const unsigned int M1_IN_2_CHANNEL = 9;
const unsigned int M2_IN_1_CHANNEL = 10;
const unsigned int M2_IN_2_CHANNEL = 11;

const int M_PWM_FREQ = 5000;
const int M_PWM_BITS = 8;
const unsigned int MAX_PWM_VALUE = 255; // Max PWM given 8 bit resolution

float METERS_PER_TICK = (3.14159 * 0.031) / 360.0;
float TURNING_RADIUS_METERS = 4.3 / 100.0; // Wheels are about 4.3 cm from pivot point

void configure_imu() {
  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      ledcWriteNote(BUZZ_CHANNEL, NOTE_C, 4);
      delay(500);
      ledcWriteNote(BUZZ_CHANNEL, NOTE_G, 4);
      delay(500);
    }
  }
  Serial.println("MPU6050 Found!");
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_1000_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
}

void read_imu(float& w_z) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  w_z = g.gyro.z;
}

void est_imu_bias(float& E_w_z, int N_samples) {
  float E_w_z_acc = 0.0;
  for (unsigned int i = 0; i < N_samples; i++) {
    float w_z;
    read_imu(w_z);
    E_w_z_acc += w_z;
    delay(5);
  }
  E_w_z = E_w_z_acc / N_samples;
}

void configure_motor_pins() {
  ledcSetup(M1_IN_1_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M1_IN_2_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M2_IN_1_CHANNEL, M_PWM_FREQ, M_PWM_BITS);
  ledcSetup(M2_IN_2_CHANNEL, M_PWM_FREQ, M_PWM_BITS);

  ledcAttachPin(M1_IN_1, M1_IN_1_CHANNEL);
  ledcAttachPin(M1_IN_2, M1_IN_2_CHANNEL);
  ledcAttachPin(M2_IN_1, M2_IN_1_CHANNEL);
  ledcAttachPin(M2_IN_2, M2_IN_2_CHANNEL);
}

// Positive means forward, negative means backwards
void set_motors_pwm(float left_pwm, float right_pwm) {
  if (isnan(left_pwm)) left_pwm = 0.0;
  if (left_pwm  >  255.0) left_pwm  =  255.0;
  if (left_pwm  < -255.0) left_pwm  = -255.0;
  if (isnan(right_pwm)) right_pwm = 0.0;
  if (right_pwm >  255.0) right_pwm =  255.0;
  if (right_pwm < -255.0) right_pwm = -255.0;

  if (left_pwm > 0) {
    ledcWrite(M1_IN_1_CHANNEL, 0);
    ledcWrite(M1_IN_2_CHANNEL, (uint32_t)(left_pwm));
  } else {
    ledcWrite(M1_IN_1_CHANNEL, (uint32_t)-left_pwm);
    ledcWrite(M1_IN_2_CHANNEL, 0);
  }

  if (right_pwm > 0) {
    ledcWrite(M2_IN_1_CHANNEL, 0);
    ledcWrite(M2_IN_2_CHANNEL, (uint32_t)(right_pwm));
  } else {
    ledcWrite(M2_IN_1_CHANNEL, (uint32_t)-right_pwm);
    ledcWrite(M2_IN_2_CHANNEL, 0);
  }
}

float update_pid(float dt, float kp, float ki, float kd,
                 float x_d, float x,
                 float& int_e, float abs_int_e_max, // last_x and int_e are updated by this function
                 float& last_x) {
  // Calculate or update intermediates
  float e = x_d - x; // Error

  // Integrate error with anti-windup
  int_e = int_e + e * dt;
  if (int_e >  abs_int_e_max) int_e =  abs_int_e_max;
  if (int_e < -abs_int_e_max) int_e = -abs_int_e_max;

  // Take the "Derivative of the process variable" to avoid derivative spikes if setpoint makes step change
  // with abuse of notation, call this de
  float de = -(x - last_x) / dt;
  last_x = x;

  float u = kp * e + ki * int_e + kd * de;
  return u;
}
void leminscate_of_bernoulli(float t, float a, float& x, float& y) {
  float sin_t = sin(t);
  float den = 1 + sin_t * sin_t;
  x = a * cos(t) / den;
  y = a * sin(t) * cos(t) / den;
}


// Signed angle from (x0, y0) to (x1, y1)
// assumes norms of these quantities are precomputed
float signed_angle(float x0, float y0, float n0, float x1, float y1, float n1) {
  float normed_dot = (x1 * x0 + y1 * y0) / (n1 * n0);
  if (normed_dot > 1.0) normed_dot = 1.0; // Possible because of numerical error
  float angle = acosf(normed_dot);

  // use cross product to find direction of rotation
  // https://en.wikipedia.org/wiki/Cross_product#Coordinate_notation
  float s3 = x0 * y1 - x1 * y0;
  if (s3 < 0) angle = -angle;

  return angle;
}

void setup() {
  // Stop the right motor by setting pin 14 low
  // this pin floats high or is pulled
  // high during the bootloader phase for some reason
  pinMode(14, OUTPUT); 
  digitalWrite(14, LOW);
  delay(100);
  
  Serial.begin(115200);
  delay(10);

  // Disable the lightbar ADC chips so they don't hold the SPI bus used by the IMU
  pinMode(ADC_1_CS, OUTPUT);
  pinMode(ADC_2_CS, OUTPUT);
  digitalWrite(ADC_1_CS, HIGH);
  digitalWrite(ADC_2_CS, HIGH);

  ledcAttachPin(BUZZ, BUZZ_CHANNEL);

  pinMode(VCC_SENSE, INPUT);

  configure_motor_pins();
  configure_imu();

//Wifi Setup - Connect to the WiFi network
  connectToWiFi(networkName, networkPswd); 
}

void loop() {


  // Create the encoder objects after the motor has
  // stopped, else some sort exception is triggered
  Encoder enc1(M1_ENC_A, M1_ENC_B);
  Encoder enc2(M2_ENC_A, M2_ENC_B);

  // Loop period
  int target_period_ms = 2; // Loop takes about 3 ms so a delay of 2 gives 200 Hz or 5ms

  // States used to calculate target velocity and heading
  float leminscate_a = 0.5; // Radius
  float leminscate_t_scale = 2.0; // speedup factor
  float x0, y0;
  leminscate_of_bernoulli(0.0, leminscate_a, x0, y0);
  float last_x, last_y;
  leminscate_of_bernoulli(-leminscate_t_scale * target_period_ms / 1000.0, leminscate_a, last_x, last_y);
  float last_dx = (x0 - last_x) / ((float)target_period_ms / 1000.0);
  float last_dy = (y0 - last_y) / ((float)target_period_ms / 1000.0);
  float last_target_v = sqrtf(last_dx * last_dx + last_dy * last_dy);
  float target_theta = 0.0; // This is an integrated quantity
  float track_target_theta = 0;
  // Motors are controlled by a position PID
  // with inputs interpreted in meters and outputs interpreted in volts
  // integral term has "anti-windup"
  // derivative term uses to derivative of process variable (wheel position)
  // instead of derivative of error in order to avoid "derivative kick"
  float kp_left = 200.0;
  float ki_left = 20.0;
  float kd_left = 20.0;
  float kf_left = 10.0;
  float target_pos_left  = 0.0;
  float last_pos_left = 0.0;
  float integral_error_pos_left = 0.0;
  float max_integral_error_pos_left = 1.0 * 8.0 / ki_left; // Max effect is the nominal battery voltage

  float kp_right = 200.0;
  float ki_right = 20.0;
  float kd_right = 20.0;
  float kf_right = 10.0;
  float last_pos_right = 0.0;
  float target_pos_right = 0.0;
  float integral_error_pos_right = 0.0;
  float max_integral_error_pos_right = 1.0 * 8.0 / ki_right; // Max effect is the nominal battery voltage

  // IMU Orientation variables
  float theta = 0.0;
  float bias_omega;
  // Gain applied to heading error when offseting target motor velocities
  // currently set to 360 deg/s compensation for 90 degrees of error
  float ktheta = (2 * 3.14159) / (90.0 * 3.14159 / 180.0);
  est_imu_bias(bias_omega, 500);// Could be expanded for more quantities

  // The real "loop()"
  // time starts from 0
  float start_t = (float)micros() / 1000000.0;
  float last_t = -target_period_ms / 1000.0; // Offset by expected looptime to avoid divide by zero
  int state = 0; //STATE 0 = INITIAL SPIN
  int k = 0;
  int j = 0;
  float t_lastpacketsent = start_t;
  // MAIN LOOP
  while (true) {
    //while wifi udp is connected? 
    //At end, if statement that disconnects UDP so we exit while(connected) loop ( if(mouse is found) disconnect UDP )
    int t_start = micros();
    int t_end = micros();
    // Get the time elapsed
    float t = ((float)micros()) / 1000000.0 - start_t;
    
    //Serial.print(t);
    float dt = ((float)(t - last_t)); // Calculate time since last update
    // Serial.print("t "); Serial.print(t);
    //Serial.print(" dt "); Serial.print(dt * 1000.0);
    last_t = t; //SEND TIME OF THETA

    // Get the distances the wheels have traveled in meters
    // positive is forward
    float pos_left  =  (float)enc1.read() * METERS_PER_TICK;
    float pos_right = -(float)enc2.read() * METERS_PER_TICK; // Take negative because right counts upwards when rotating backwards
  
    // TODO Battery voltage compensation, the voltage sense on my mouse is broken for some reason
    // int counts = analogRead(VCC_SENSE);
    // float battery_voltage = counts * ADC_COUNTS_TO_VOLTS;
    // if (battery_voltage <= 0) Serial.println("BATTERY INVALID");
  
    // Read IMU and update estimate of heading
    // positive is counter clockwise
    float omega;
    read_imu(omega); // Could be expanded to read more things
    omega -= bias_omega; // Remove the constant bias measured in the beginning
    theta = theta + omega * dt;
   

    // Calculate target forward velocity and target heading to track the leminscate trajectory
    // of 0.5 meter radius
    float x, y;
   leminscate_of_bernoulli(leminscate_t_scale * t, leminscate_a, x, y);
    float dx = (x - last_x) / dt;
    float dy = (y - last_y) / dt;
    // Compute the change in heading using the normalized dot product between the current and last velocity vector
    // using this method instead of atan2 allows easy smooth handling of angles outsides of -pi / pi at the cost of
    // a slow drift defined by numerical precision
   // float target_omega = signed_angle(last_dx, last_dy, last_target_v, dx, dy, target_v) / dt;

  
    float maxtheta; //store only 1 maxtheta, NEED ARRAY OF MAXTHETA maxtheta[0] will be 0
    float target_v = 0;//sqrtf(dx * dx + dy * dy); // SPEED OF ROBOT
    float target_omega = 0;//= signed_angle(last_dx, last_dy, last_target_v, dx, dy, target_v) / dt; 
    int stopped = true;
    //STATE MACHINE!!!
    if(state == 0){ //Initial spin
     // target_theta = target_theta;
      target_omega = M_PI; //increment by pi/4 CHANGED TO pi/2 FOR TEST
      // target_theta = target_theta + target_omega * dt;
      // Serial.print("test theta ="); Serial.println(target_theta);
       //SEND PACKET
      stopped = false;
      if(t - t_lastpacketsent > .03) { //good speed, maybe little slower
        
        target_theta = target_theta + target_omega * dt;
        track_target_theta = track_target_theta + target_omega*dt; //so that we can trakc 0-2pi every time (even when targettheta > 2pi; we don't have to reset targettheta)
        //Serial.print("test theta ="); Serial.println(target_theta);
        //track_target_theta = target_theta;
        udp.beginPacket(udpAddress,mouse1_udpPort);
        udp.printf("%f %f %f", target_theta, t, track_target_theta); //prints theta and corresponding time t to Jetson SHOULD THESE BE %lu
        udp.endPacket();
        Serial.print("theta = "); Serial.println(target_theta);
        t_lastpacketsent = t;
        
        if(track_target_theta>=2*M_PI) { //IF 360deg IS COMPLETE
          udp.beginPacket(udpAddress,mouse1_udpPort);
          udp.printf("%f %f %f", target_theta, t, track_target_theta); //prints theta and corresponding time t to Jetson SHOULD THESE BE %lu
          udp.endPacket();
          Serial.print("GOT HERE");
          //target_theta = 0;
          state = 1; // GO TO NEXT STATE
        }
      }
      else target_theta = target_theta;
    } //if state 0 (initial spin)

    else if(state ==1){ //STATE 1= RECEIVE PACKET, SPIN TO THAT THETA
        //RECEIVE PACKET
        stopped = false;
        int packetSize = udp.parsePacket();
        if(packetSize >= sizeof(float)){
          udp.read((char*)&maxtheta, sizeof(maxtheta)); //need the (char*)
          udp.flush();
          // target_theta = maxtheta; //ASSIGN THETA TO MAX AMP THETA
          Serial.printf("maxtheta is %f\n", maxtheta); 
          //target_omega = maxtheta; //set angle to theta with max volume
          //target_theta = target_theta + target_omega * dt;
          target_theta = maxtheta;
          Serial.printf("targettheta %f\n", target_theta);
          state = 2; //GO TO NEXT STATE
        } //if (send UDP)
        
    }

    else if(state == 2){ //MOVE FORWARD '2' SECONDS, SEND SOMETHING TO MAKE PYTHON RECORD SOUND AGAIN AND CROSS CORRELATE 3 MICS (GO TO SPECIFIC PART OF PYTHON CODE)

      target_theta = target_theta; //PROBABLY DON'T NEED JUST WANNA MAKE SURE IT STAYS AT THIS ANGLE
      
      //Serial.print("state 2");
      if(k< 900){ //move forward until k=
        stopped = false;
        target_v = .1;//Move forward
        k++;
      }
      else{target_v = 0; stopped = true; track_target_theta = 0; state = 3; k = 0;} //state = 3; (mouse2) state = 0; (mouse1)
    }

// MOUSE 2 ONLY
    else if(state == 3){ 
      target_theta = target_theta;
      stopped = true;
      if(j < 1000){ //Stay stopped
        target_v = 0;//stop robot
        target_theta = target_theta;
        // stopped = true;
        j++;
      }
      else{ //receive packet
        // stopped = true;
          j = 0; //target_theta = target_theta;
          state = 4;
         // CHANGE STATE
      }
      //SEND PACKET TO TELL MICS TO START RECORDING AGAIN
      if(t - t_lastpacketsent > .02) {
        udp.beginPacket(udpAddress,mouse1_udpPort);
        udp.printf("%d", state); //send state
        udp.endPacket();
        t_lastpacketsent = t;
        target_theta = target_theta;
      }
    

    }

    else if(state == 4){ //RECEIVE PACKET
      int packetSize = udp.parsePacket();
      if(packetSize >= sizeof(float)){
        udp.read((char*)&maxtheta, sizeof(maxtheta)); //need the (char*)
        udp.flush();
        // target_theta = maxtheta; //ASSIGN THETA TO MAX AMP THETA
        Serial.printf("message received: target_omega is %f\n", maxtheta); 
        //target_omega = maxtheta; //set angle to theta with max volume
        //target_theta = target_theta + target_omega * dt;
        target_omega = maxtheta;
        target_theta = target_theta + target_omega*dt;
        Serial.printf("target_theta %f\n", target_theta);
        state = 2; //GO TO NEXT STATE
        stopped = false;
      }
    }
    //target_theta = target_theta + target_omega * dt; //should be in each state

  //only send data when connected

//   //AFTER 360 SPIN IS FINISHED
//   //ROBOT COMES TO STOP

//   target_theta = maxtheta[1]; //ASSIGN THETA TO MAX AMP THETA






    last_x = x;
    last_y = y;
    last_dx = dx;
    last_dy = dy;
    last_target_v = target_v;
  
    // Calculate target motor speeds from target forward speed and target heading
    // Could also include target path length traveled and target angular velocity
    float error_theta_z = target_theta - theta;
    float requested_v = target_v;
    float requested_w = ktheta * error_theta_z;

    float target_v_left  = requested_v - TURNING_RADIUS_METERS * requested_w;
    float target_v_right = requested_v + TURNING_RADIUS_METERS * requested_w;
    target_pos_left  = target_pos_left  + dt * target_v_left;
    target_pos_right = target_pos_right + dt * target_v_right;



    // Left motor position PID
    float left_voltage = update_pid(dt, kp_left, ki_left, kd_left,
                                    target_pos_left, pos_left,
                                    integral_error_pos_left, max_integral_error_pos_left,
                                    last_pos_left);
    left_voltage = left_voltage + kf_left * target_v_left;
    float left_pwm = (float)MAX_PWM_VALUE * (left_voltage / 8.0); // TODO use actual battery voltage

    // Right motor position PID
    float right_voltage = update_pid(dt, kp_right, ki_right, kd_right,
                                     target_pos_right, pos_right,
                                     integral_error_pos_right, max_integral_error_pos_right,
                                     last_pos_right);
    left_voltage = right_voltage + kf_right * target_v_right;
    float right_pwm = (float)MAX_PWM_VALUE * (right_voltage / 8.0); // TODO use actual battery voltage
  if(stopped == true){ //Robot Stopped
    set_motors_pwm(0,0);
  }
  else{
   set_motors_pwm(left_pwm, right_pwm);
  }
  //set_motors_pwm(left_pwm, right_pwm);
    // Serial.println();
    delay(target_period_ms);
  } //while(connected == true ) loop
} //void loop()
