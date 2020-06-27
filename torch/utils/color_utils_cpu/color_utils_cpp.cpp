#include <torch/extension.h>

#include <vector>
#include <cmath>
//#include <fstream>


struct vec3f {
	vec3f() {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}
	vec3f(float x_, float y_, float z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	inline vec3f operator+(const vec3f& other) const {
		return vec3f(x+other.x, y+other.y, z+other.z);
	}
	inline vec3f operator-(const vec3f& other) const {
		return vec3f(x-other.x, y-other.y, z-other.z);
	}
	inline vec3f operator*(float val) const {
		return vec3f(x*val, y*val, z*val);
	}
	inline void operator+=(const vec3f& other) {
		x += other.x;
		y += other.y;
		z += other.z;
	}
	static float distSq(const vec3f& v0, const vec3f& v1) {
		return ((v0.x-v1.x)*(v0.x-v1.x) + (v0.y-v1.y)*(v0.y-v1.y) + (v0.z-v1.z)*(v0.z-v1.z));
	}
	float x;
	float y;
	float z;
};
inline vec3f operator*(float s, const vec3f& v) {
	return v * s;
}

//vec3f convert_rgb_to_hsv_vec3(const float r, const float g, const float b) {
//	vec3f hsv(0.0f, 0.0f, 0.0f);
//	const float mn = std::min(r, std::min(g, b));
//	const float mx = std::max(r, std::max(g, b));
//	hsv.z = mx;
//	const float delta = mx - mn;
//	if (r + g + b > 0.01f) {
//		hsv.y = delta / mx;
//	}
//	if (delta > 0.001f) {
//		if (mx == r) {
//			hsv.x = 60.0f * (0.0f + (g - b) / delta);
//		}
//		else if (mx == g) {
//			hsv.x = 60.0f * (2.0f + (b - r) / delta);
//		}
//		else if (mx == b) {
//			hsv.x = 60.0f * (4.0f + (r - g) / delta);
//		}
//		if (hsv.x < 0) hsv.x += 360.0f;
//	}
//	return hsv;
//}
//vec3f convert_rgb_to_hsv_vec3(const vec3f& rgb) {
//	return convert_rgb_to_hsv_vec3(rgb.x, rgb.y, rgb.z);
//}
void convert_rgb_to_hsv_vec3(const float r, const float g, const float b, float& h, float& s, float& v) {
	const float mn = std::min(r, std::min(g, b));
	const float mx = std::max(r, std::max(g, b));
	v = mx;
	const float delta = mx - mn;
	if (r + g + b > 0.01f) {
		s = delta / mx;
	}
	else {
		s = 0.0f;
	}
	if (delta > 0.001f) {
		if (mx == r) {
			h = 60.0f * (0.0f + (g - b) / delta);
		}
		else if (mx == g) {
			h = 60.0f * (2.0f + (b - r) / delta);
		}
		else if (mx == b) {
			h = 60.0f * (4.0f + (r - g) / delta);
		}
		if (h < 0) h += 360.0f;
	}
	else {
		h = 0.0f;
	}
}

//vec3f convert_hsv_to_rgb_vec3(const float h, const float s, const float v) {
//	const float hd = h / 60.0f;
//	const int hi = (int)std::floor(hd);
//	const float f = hd - (float)hi;
//	const float p = v*(1.0f - s);
//	const float q = v*(1.0f - s*f);
//	const float t = v*(1.0f - s*(1.0f-f));
//	vec3f rgb(0.0f, 0.0f, 0.0f);
//	if (hi == 0 || hi == 6) {
//		rgb.x = v;
//		rgb.y = t;
//		rgb.z = p;
//	}
//	else if (hi == 1) {
//		rgb.x = q;
//		rgb.y = v;
//		rgb.z = p;
//	}
//	else if (hi == 2) {
//		rgb.x = p;
//		rgb.y = v;
//		rgb.z = t;
//	}
//	else if (hi == 3) {
//		rgb.x = p;
//		rgb.y = q;
//		rgb.z = v;
//	}
//	else if (hi == 4) {
//		rgb.x = t;
//		rgb.y = p;
//		rgb.z = v;
//	}
//	else if (hi == 5) {
//		rgb.x = v;
//		rgb.y = p;
//		rgb.z = q;
//	}	
//	return rgb;
//}
//vec3f convert_hsv_to_rgb_vec3(const vec3f& hsv) {
//	return convert_hsv_to_rgb_vec3(hsv.x, hsv.y, hsv.z);
//}
void convert_hsv_to_rgb_vec3(const float h, const float s, const float v, float& r, float& g, float& b) {
	const float hd = h / 60.0f;
	const int hi = (int)std::floor(hd);
	const float f = hd - (float)hi;
	const float p = v*(1.0f - s);
	const float q = v*(1.0f - s*f);
	const float t = v*(1.0f - s*(1.0f-f));
	if (hi == 0 || hi == 6) {
		r = v;
		g = t;
		b = p;
	}
	else if (hi == 1) {
		r = q;
		g = v;
		b = p;
	}
	else if (hi == 2) {
		r = p;
		g = v;
		b = t;
	}
	else if (hi == 3) {
		r = p;
		g = q;
		b = v;
	}
	else if (hi == 4) {
		r = t;
		g = p;
		b = v;
	}
	else if (hi == 5) {
		r = v;
		g = p;
		b = q;
	}
}

void convert_rgb_to_lab_vec3(float r, float g, float b, float& l_, float& a_, float& b_) {
	//printf("convert_rgb_to_lab_vec3(%f, %f, %f)\n", r, g, b); fflush(stdout);
	r = (r > 0.0405f) ? std::pow((r + 0.055f) / 1.055f, 2.4f) : r/12.92f;
	g = (g > 0.0405f) ? std::pow((g + 0.055f) / 1.055f, 2.4f) : g/12.92f;
	b = (b > 0.0405f) ? std::pow((b + 0.055f) / 1.055f, 2.4f) : b/12.92f;
	float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
	float y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
	float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
	//printf("\t(x,y,z) = (%f, %f, %f)\n", x, y, z); fflush(stdout);
	
	x /= 0.95047f;
	//y /= 1.0f;
	z /= 1.08883f;
	
	x = (x > 0.008856f) ? std::pow(x, 1.0f/3.0f) : 7.787f * x + 16.0f / 116.0f;
	y = (y > 0.008856f) ? std::pow(y, 1.0f/3.0f) : 7.787f * y + 16.0f / 116.0f;
	z = (z > 0.008856f) ? std::pow(z, 1.0f/3.0f) : 7.787f * z + 16.0f / 116.0f;
	
	l_ = 116.0f*y-16.0f;
	a_ = 500.0f*(x-y);
	b_ = 200.0f*(y-z);
}
void convert_lab_to_rgb_vec3(float l, float a, float b, float& r_, float& g_, float& b_) {
    float y = (l + 16.0f) / 116.0f;
    float x = (a / 500.0f) + y;
    float z = y - (b / 200.0f);
	if (z < 0) z = 0; // invalid
	x = (x > 0.2068966f) ? std::pow(x, 3.0f) : (x - 16.0f / 116.0f) / 7.787f;
	y = (y > 0.2068966f) ? std::pow(y, 3.0f) : (y - 16.0f / 116.0f) / 7.787f;
	z = (z > 0.2068966f) ? std::pow(z, 3.0f) : (z - 16.0f / 116.0f) / 7.787f;
    x *= 0.95047f;
    //y *= 1.0f;
    z *= 1.08883f;
	r_ = 3.2405f * x -1.5372f * y -0.4985f * z;
	g_ = -0.9693f * x + 1.8760f * y + 0.0416f * z;
	b_ = 0.0556f * x -0.2040f *  y + 1.0573f * z;
	r_ = (r_ > 0.0031308f) ? 1.055f * std::pow(r_, 1.0f / 2.4f) - 0.055f : r_ * 12.92f;
	g_ = (g_ > 0.0031308f) ? 1.055f * std::pow(g_, 1.0f / 2.4f) - 0.055f : g_ * 12.92f;
	b_ = (b_ > 0.0031308f) ? 1.055f * std::pow(b_, 1.0f / 2.4f) - 0.055f : b_ * 12.92f;
}


//could actually just do this in place ? w/o allocating new array

/*void convert_rgb_to_hsv(
    at::Tensor rgb,
    at::Tensor hsv,
	bool bNormalized01) {
	
	auto rgb_accessor = rgb.accessor<float,2>();
	auto hsv_accessor = hsv.accessor<float,2>();
	const long num = rgb.size(0);
	for (unsigned int i = 0; i < num; i++) {
		vec3f hsv = convert_rgb_to_hsv_vec3(rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2]);
		hsv_accessor[i][0] = bNormalized01 ? hsv.x/360.0f : hsv.x;
		hsv_accessor[i][1] = hsv.y;
		hsv_accessor[i][2] = hsv.z;
	}
}*/
void convert_rgb_to_hsv(
    at::Tensor rgb,
    at::Tensor hsv,
	bool bNormalized01) {
	
	auto rgb_accessor = rgb.accessor<float,2>();
	auto hsv_accessor = hsv.accessor<float,2>();
	const long num = rgb.size(0);
	for (unsigned int i = 0; i < num; i++) {
		convert_rgb_to_hsv_vec3(rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2], hsv_accessor[i][0], hsv_accessor[i][1], hsv_accessor[i][2]);
		if (bNormalized01) hsv_accessor[i][0] /= 360.0f;
	}
}

void convert_hsv_to_rgb(
    at::Tensor hsv,
    at::Tensor rgb,
	bool bNormalized01) {
	
	auto hsv_accessor = hsv.accessor<float,2>();
	auto rgb_accessor = rgb.accessor<float,2>();
	const long num = hsv.size(0);
	for (unsigned int i = 0; i < num; i++) {
		convert_hsv_to_rgb_vec3(bNormalized01 ? hsv_accessor[i][0]*360.0f : hsv_accessor[i][0], hsv_accessor[i][1], hsv_accessor[i][2], rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2]);
	}
}


void convert_rgb_to_lab(
    at::Tensor rgb,
    at::Tensor lab,
	bool bNormalized01) {
	
	auto rgb_accessor = rgb.accessor<float,2>();
	auto lab_accessor = lab.accessor<float,2>();
	const long num = rgb.size(0);
	for (unsigned int i = 0; i < num; i++) {
		convert_rgb_to_lab_vec3(rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2], lab_accessor[i][0], lab_accessor[i][1], lab_accessor[i][2]);
		if (bNormalized01) {
			lab_accessor[i][0] *= 0.01f;
			lab_accessor[i][1] = (lab_accessor[i][1]+100.0f)*0.005f;
			lab_accessor[i][2] = (lab_accessor[i][2]+100.0f)*0.005f;
		}
	}
}

void convert_lab_to_rgb(
    at::Tensor lab,
    at::Tensor rgb,
	bool bNormalized01) {
	
	auto lab_accessor = lab.accessor<float,2>();
	auto rgb_accessor = rgb.accessor<float,2>();
	const long num = lab.size(0);
	for (unsigned int i = 0; i < num; i++) {
		if (bNormalized01) convert_lab_to_rgb_vec3(lab_accessor[i][0]*100.0f, lab_accessor[i][1]*200.0f-100.0f, lab_accessor[i][2]*200.0f-100.0f, rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2]);
		else convert_lab_to_rgb_vec3(lab_accessor[i][0], lab_accessor[i][1], lab_accessor[i][2], rgb_accessor[i][0], rgb_accessor[i][1], rgb_accessor[i][2]);
	}
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("convert_rgb_to_hsv", &convert_rgb_to_hsv, "RGB->HSV conversion");
  m.def("convert_hsv_to_rgb", &convert_hsv_to_rgb, "HSV->RGB conversion");
  m.def("convert_rgb_to_lab", &convert_rgb_to_lab, "RGB->LAB conversion");
  m.def("convert_lab_to_rgb", &convert_lab_to_rgb, "LAB->RGB conversion");
}
