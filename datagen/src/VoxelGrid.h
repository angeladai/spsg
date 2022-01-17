#pragma once

#include "mLibInclude.h"


struct Voxel {
	Voxel() {
		sdf = -std::numeric_limits<float>::infinity();
		freeCtr = 0;
		color = vec3uc(0, 0, 0);
		weight = 0;
	}

	float			sdf;
	unsigned int	freeCtr;
	vec3uc			color;
	uchar			weight;
};

class VoxelGrid : public Grid3 < Voxel >
{
public:

	VoxelGrid(const vec3l& dim, const mat4f& worldToGrid, float voxelSize, const OBB3f& gridSceneBounds, float depthMin, float depthMax) : Grid3(dim.x, dim.y, dim.z) {
		m_voxelSize = voxelSize;
		m_depthMin = depthMin;
		m_depthMax = depthMax;
		m_worldToGrid = worldToGrid;
		m_gridToWorld = m_worldToGrid.getInverse();
		m_sceneBoundsGrid = gridSceneBounds;

		m_trunaction = m_voxelSize * 3.0f;
		m_truncationScale = m_voxelSize;
		m_weightUpdate = 1;
	}

	~VoxelGrid() {

	}

	void reset() {
#pragma omp parallel for
		for (int i = 0; i < (int)getNumElements(); i++) {
			getData()[i] = Voxel();
		}
	}

	void integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const ColorImageR8G8B8& color);
	void integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& mask);
	//void integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& semantics);

	//! normalizes the SDFs (divides by the voxel size)
	void normalizeSDFs(float factor = -1.0f) {
		if (factor < 0) factor = 1.0f / m_voxelSize;
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					Voxel& v = (*this)(i, j, k);
					if (v.sdf != -std::numeric_limits<float>::infinity() && v.sdf != 0.0f) {
						v.sdf *= factor;
					}

				}
			}
		}
		m_voxelSize *= factor;
	}

	//! returns all the voxels on the isosurface
	std::vector<Voxel> getSurfaceVoxels(unsigned int weightThresh, float sdfThresh) const {

		std::vector<Voxel> res;
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					const Voxel& v = (*this)(i, j, k);
					if (v.weight >= weightThresh && std::abs(v.sdf) < sdfThresh) {
						res.push_back(v);
					}
				}
			}
		}

		return res;
	}

	BinaryGrid3 toBinaryGridFree(unsigned int freeThresh) const {
		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).freeCtr >= freeThresh) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}

	BinaryGrid3 toBinaryGridOccupied(unsigned int weightThresh, float sdfThresh) const {

		BinaryGrid3 res(getDimX(), getDimY(), getDimZ());
		for (size_t k = 0; k < getDimZ(); k++) {
			for (size_t j = 0; j < getDimY(); j++) {
				for (size_t i = 0; i < getDimX(); i++) {

					if ((*this)(i, j, k).weight >= weightThresh && std::abs((*this)(i, j, k).sdf) < sdfThresh) {
						//if ((*this)(i, j, k).weight >= weightThresh && (*this)(i, j, k).sdf < sdfThresh && (*this)(i, j, k).sdf >= 0) {
						res.setVoxel(i, j, k);
					}
				}
			}
		}
		return res;
	}


	TriMeshf computeSemanticsMesh(float sdfThresh) const;

	void saveToFile(const std::string& filename, bool bSaveSparse, bool bSaveColors = false, float truncationFactor = -1.0f) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&m_voxelSize, sizeof(float));
		ofs.write((const char*)m_worldToGrid.getData(), sizeof(mat4f));
		std::vector<vec3uc> colorValues;
		if (bSaveSparse) {
			if (truncationFactor <= 0) truncationFactor = m_trunaction / m_voxelSize;
			std::vector<vec3ui> locations;
			std::vector<float> sdfvalues;
			for (unsigned int z = 0; z < dimZ; z++) {
				for (unsigned int y = 0; y < dimY; y++) {
					for (unsigned int x = 0; x < dimX; x++) {
						const Voxel& v = (*this)(x, y, z);
						if (std::fabs(v.sdf) <= truncationFactor*m_voxelSize) {
							locations.push_back(vec3ui(x, y, z));
							sdfvalues.push_back(v.sdf);

							if (bSaveColors) colorValues.push_back(v.color);
						}
					} // x
				} // y
			} // z
			UINT64 num = (UINT64)locations.size();
			ofs.write((const char*)&num, sizeof(UINT64));
			ofs.write((const char*)locations.data(), sizeof(vec3ui)*locations.size());
			ofs.write((const char*)sdfvalues.data(), sizeof(float)*sdfvalues.size());
		}
		else {
			//dense data
			std::vector<float> values(getNumElements());
			if (bSaveColors) colorValues.resize(getNumElements());
			for (unsigned int i = 0; i < getNumElements(); i++) {
				const Voxel& v = getData()[i];
				values[i] = v.sdf;
				if (bSaveColors) colorValues[i] = v.color;
			}
			ofs.write((const char*)values.data(), sizeof(float)*values.size());
		}
		ofs.close();
		if (bSaveColors) {
			std::ofstream ofsMask(util::removeExtensions(filename) + ".colors", std::ios::binary);
			ofsMask.write((const char*)&dimX, sizeof(UINT64));
			ofsMask.write((const char*)&dimY, sizeof(UINT64));
			ofsMask.write((const char*)&dimZ, sizeof(UINT64));
			if (bSaveSparse) {
				UINT64 num = (UINT64)colorValues.size();
				ofsMask.write((const char*)&num, sizeof(UINT64));
				ofsMask.write((const char*)colorValues.data(), sizeof(vec3uc)*colorValues.size());
			}
			else {
				ofsMask.write((const char*)colorValues.data(), sizeof(vec3uc)*colorValues.size());
			}
			ofsMask.close();
		}
	}
	void loadFromFile(const std::string& filename, bool bLoadSparse, bool bLoadColors) {
		std::ifstream ifs(filename, std::ios::binary);
		//metadata
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		ifs.read((char*)&m_voxelSize, sizeof(float));
		ifs.read((char*)m_worldToGrid.getData(), sizeof(mat4f));
		m_gridToWorld = m_worldToGrid.getInverse();
		//dense data
		allocate(dimX, dimY, dimZ);
		if (bLoadSparse) {
			UINT64 num;
			ifs.read((char*)&num, sizeof(UINT64));
			//std::cout << filename << ": " << num << std::endl;
			std::vector<vec3ui> locations(num);
			std::vector<float> sdfvalues(num);
			ifs.read((char*)locations.data(), sizeof(vec3ui)*locations.size());
			ifs.read((char*)sdfvalues.data(), sizeof(float)*sdfvalues.size());
			std::vector<vec3uc> colors;
			if (bLoadColors) {
				const std::string colorFile = util::removeExtensions(filename) + ".colors";
				if (!util::fileExists(colorFile)) throw MLIB_EXCEPTION("color file " + colorFile + " does not exist!");
				std::ifstream ifsMask(colorFile, std::ios::binary);
				UINT64 colorX, colorY, colorZ, numColor;
				ifsMask.read((char*)&colorX, sizeof(UINT64));
				ifsMask.read((char*)&colorY, sizeof(UINT64));
				ifsMask.read((char*)&colorZ, sizeof(UINT64));
				if (dimX != colorX || dimY != colorY || dimZ != colorZ) throw MLIB_EXCEPTION("ERROR: dim grid (" + std::to_string(dimX) + ", " + std::to_string(dimY) + "," + std::to_string(dimZ) + ") != #color (" + std::to_string(colorX) + ", " + std::to_string(colorY) + "," + std::to_string(colorZ) + ")");
				ifsMask.read((char*)&numColor, sizeof(UINT64));
				if (num != numColor) throw MLIB_EXCEPTION("ERROR: #vals (" + std::to_string(num) + ") != #color (" + std::to_string(numColor) + ")");
				colors.resize(numColor);
				ifsMask.read((char*)colors.data(), sizeof(vec3uc)*numColor);
				ifsMask.close();
			}
			for (unsigned int i = 0; i < locations.size(); i++) {
				Voxel& v = (*this)(locations[i]);
				v.sdf = sdfvalues[i];
				if (v.sdf > -m_voxelSize) v.weight = 1; //just for vis purposes
				if (bLoadColors) v.color = colors[i];
			}
		}
		else {
			std::vector<float> values(getNumElements());
			ifs.read((char*)values.data(), sizeof(float)*values.size());
			std::vector<vec3uc> colors;
			if (bLoadColors) {
				const std::string colorFile = util::removeExtensions(filename) + ".colors";
				if (!util::fileExists(colorFile)) throw MLIB_EXCEPTION("color file " + colorFile + " does not exist!");
				std::ifstream ifsMask(colorFile, std::ios::binary);
				UINT64 colorX, colorY, colorZ, numColor;
				ifsMask.read((char*)&colorX, sizeof(UINT64));
				ifsMask.read((char*)&colorY, sizeof(UINT64));
				ifsMask.read((char*)&colorZ, sizeof(UINT64));
				if (dimX != colorX || dimY != colorY || dimZ != colorZ) throw MLIB_EXCEPTION("ERROR: dim grid (" + std::to_string(dimX) + ", " + std::to_string(dimY) + "," + std::to_string(dimZ) + ") != #mask (" + std::to_string(colorX) + ", " + std::to_string(colorY) + "," + std::to_string(colorZ) + ")");
				ifsMask.read((char*)&numColor, sizeof(UINT64));
				if (getNumElements() != numColor) throw MLIB_EXCEPTION("ERROR: #vals (" + std::to_string(getNumElements()) + ") != #color (" + std::to_string(numColor) + ")");
				colors.resize(numColor);
				ifsMask.read((char*)colors.data(), sizeof(vec3uc)*numColor);
				ifsMask.close();
			}
			for (unsigned int i = 0; i < getNumElements(); i++) {
				Voxel& v = getData()[i];
				v.sdf = values[i];
				if (v.sdf >= -m_voxelSize) v.weight = 1;
				if (bLoadColors) v.color = colors[i];
			}
		}
		ifs.close();
	}
	void saveSparseColorsTo(const std::string& filename, float truncationFactor = -1.0f) const {
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		//std::vector<vec3ui> locations;
		std::vector<vec3uc> colorValues;
		if (truncationFactor <= 0) truncationFactor = m_trunaction / m_voxelSize;
		for (unsigned int z = 0; z < dimZ; z++) {
			for (unsigned int y = 0; y < dimY; y++) {
				for (unsigned int x = 0; x < dimX; x++) {
					const Voxel& v = (*this)(x, y, z);
					if (std::fabs(v.sdf) <= truncationFactor*m_voxelSize) {
						//locations.push_back(vec3ui(x, y, z));
						colorValues.push_back(v.color);
					}
				} // x
			} // y
		} // z
		std::cout << filename << ": " << colorValues.size() << std::endl;
		std::ofstream ofs(filename, std::ios::binary);
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		UINT64 num = (UINT64)colorValues.size();
		ofs.write((const char*)&num, sizeof(UINT64));
		ofs.write((const char*)colorValues.data(), sizeof(vec3uc)*colorValues.size());
		ofs.close();
	}
	void loadSparseColorsFrom(const std::string& filename) {
		std::vector<vec3ui> locations;
		std::vector<vec3uc> colors;
		{
			std::ifstream ifs(util::removeExtensions(filename) + ".sdf", std::ios::binary);
			//metadata
			UINT64 dimX, dimY, dimZ; float voxelSize; mat4f worldToGrid;
			ifs.read((char*)&dimX, sizeof(UINT64));
			ifs.read((char*)&dimY, sizeof(UINT64));
			ifs.read((char*)&dimZ, sizeof(UINT64));
			ifs.read((char*)&voxelSize, sizeof(float));
			ifs.read((char*)worldToGrid.getData(), sizeof(mat4f));
			UINT64 num;
			ifs.read((char*)&num, sizeof(UINT64));
			locations.resize(num);
			ifs.read((char*)locations.data(), sizeof(vec3ui)*locations.size());
			ifs.close();
		}
		{
			std::ifstream ifs(filename, std::ios::binary);
			UINT64 colorX, colorY, colorZ, numColor;
			ifs.read((char*)&colorX, sizeof(UINT64));
			ifs.read((char*)&colorY, sizeof(UINT64));
			ifs.read((char*)&colorZ, sizeof(UINT64));
			ifs.read((char*)&numColor, sizeof(UINT64));
			if (locations.size() != numColor) throw MLIB_EXCEPTION("ERROR: #vals (" + std::to_string(locations.size()) + ") != #color (" + std::to_string(numColor) + ")");
			colors.resize(numColor);
			ifs.read((char*)colors.data(), sizeof(vec3uc)*numColor);
			ifs.close();
		}
		for (unsigned int i = 0; i < locations.size(); i++) {
			if (!isValidCoordinate(locations[i])) continue;
			Voxel& v = (*this)(locations[i]);
			v.color = colors[i];
		}
	}


	void saveKnownToFile(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		ofs.write((const char*)&m_voxelSize, sizeof(float));
		ofs.write((const char*)m_worldToGrid.getData(), sizeof(mat4f));
		std::vector<unsigned char> known(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			const Voxel& v = getData()[i];
			if (v.sdf < -m_voxelSize) known[i] = (unsigned char)std::max(2, std::min(255, (int)(-v.sdf / m_voxelSize) + 1));
			//if (v.sdf < -m_voxelSize)  known[i] = 2; // unknown
			else if (v.sdf <= m_voxelSize)  known[i] = 1; // known occ
			else  known[i] = 0; // known empty
		}
		ofs.write((const char*)known.data(), sizeof(unsigned char)*known.size());
		ofs.close();
	}

	void saveSemanticsToFile(const std::string& filename) const {
		std::ofstream ofs(filename, std::ios::binary);
		//metadata
		UINT64 dimX = getDimX(), dimY = getDimY(), dimZ = getDimZ();
		ofs.write((const char*)&dimX, sizeof(UINT64));
		ofs.write((const char*)&dimY, sizeof(UINT64));
		ofs.write((const char*)&dimZ, sizeof(UINT64));
		//dense data
		std::vector<unsigned short> values(getNumElements());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			const Voxel& v = getData()[i];
			values[i] = (unsigned short)v.color[0];
		}
		ofs.write((const char*)values.data(), sizeof(unsigned short)*values.size());
		ofs.close();
	}
	//! just for debugging
	void loadSemanticsFromFile(const std::string& filename) {
		std::ifstream ifs(filename, std::ios::binary);
		//metadata
		UINT64 dimX, dimY, dimZ;
		ifs.read((char*)&dimX, sizeof(UINT64));
		ifs.read((char*)&dimY, sizeof(UINT64));
		ifs.read((char*)&dimZ, sizeof(UINT64));
		if (getDimX() != dimX || getDimY() != dimY || getDimZ() != dimZ)
			throw MLIB_EXCEPTION("[loadSemanticsFromFile] mismatch dimensions");
		//dense data
		std::vector<unsigned short> values(getNumElements());
		ifs.read((char*)values.data(), sizeof(unsigned short)*values.size());
		for (unsigned int i = 0; i < getNumElements(); i++) {
			Voxel& v = getData()[i];
			v.color[0] = (unsigned char)values[i];
		}
		ifs.close();
	}

	mat4f getGridToWorld() const {
		return m_gridToWorld;
	}

	mat4f getWorldToGrid() const {
		return m_worldToGrid;
	}

	void setVoxelSize(float v) {
		m_voxelSize = v;
	}
	void setWorldToGrid(const mat4f& worldToGrid) {
		m_worldToGrid = worldToGrid;
		m_gridToWorld = worldToGrid.getInverse();
	}

	vec3i worldToVoxel(const vec3f& p) const {
		return math::round((m_worldToGrid * p));
	}

	vec3f worldToVoxelFloat(const vec3f& p) const {
		return (m_worldToGrid * p);
	}

	vec3f voxelToWorld(vec3i& v) const {
		return m_gridToWorld * (vec3f(v));
	}

	float getVoxelSize() const {
		return m_voxelSize;
	}


	bool trilinearInterpolationSimpleFastFast(const vec3f& pos, float& dist, vec3uc& color) const {
		const float oSet = m_voxelSize;
		const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
		vec3f weight = frac(worldToVoxelFloat(pos));

		dist = 0.0f;
		vec3f colorFloat = vec3f(0.0f, 0.0f, 0.0f);

		//const bool bDebug = true;// vec3f::dist(pos, vec3f(130.5f, 25.5f, 3.5f)) < 1;
		//Voxel v; vec3f vColor;
		//v = getVoxel(posDual + vec3f(0.0f, 0.0f, 0.0f)); 
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(0.0f, 0.0f, 0.0f)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		//v = getVoxel(posDual + vec3f(oSet, 0.0f, 0.0f));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(oSet, 0.0f, 0.0f)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		//v = getVoxel(posDual + vec3f(0.0f, oSet, 0.0f));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(0.0f, oSet, 0.0f)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		//v = getVoxel(posDual + vec3f(0.0f, 0.0f, oSet));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(0.0f, 0.0f, oSet)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		//v = getVoxel(posDual + vec3f(oSet, oSet, 0.0f));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(oSet, oSet, 0.0f)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		//v = getVoxel(posDual + vec3f(0.0f, oSet, oSet));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(0.0f, oSet, oSet)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		//v = getVoxel(posDual + vec3f(oSet, 0.0f, oSet));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(oSet, 0.0f, oSet)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		//v = getVoxel(posDual + vec3f(oSet, oSet, oSet));
		//if (bDebug) std::cout << "trilerp(" << pos << ") -> (" << (posDual + vec3f(oSet, oSet, oSet)) << ") = (" << v.sdf << ", " << (int)v.weight << std::endl;
		//if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		Voxel v; vec3f vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v.sdf; colorFloat += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, 0.0f)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *(1.0f - weight.z)*v.sdf; colorFloat += weight.x *	   weight.y *(1.0f - weight.z)*vColor;
		v = getVoxel(posDual + vec3f(0.0f, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += (1.0f - weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat += (1.0f - weight.x)*	   weight.y *	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, 0.0f, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *(1.0f - weight.y)*	   weight.z *v.sdf; colorFloat += weight.x *(1.0f - weight.y)*	   weight.z *vColor;
		v = getVoxel(posDual + vec3f(oSet, oSet, oSet)); if (v.weight == 0) return false;		   vColor = vec3f(v.color.x, v.color.y, v.color.z); dist += weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat += weight.x *	   weight.y *	   weight.z *vColor;

		color = vec3uc(math::round(colorFloat.x), math::round(colorFloat.y), math::round(colorFloat.z));//v.color;

		return true;
	}

	//! propagates distance values
	void improveSDF(unsigned int numIter, uchar weightThresh = 0, bool bKeepSigns = false) {
		const int radius = 2; // for weight filtering
		for (unsigned int iter = 0; iter < numIter; iter++) {
			bool hasUpdate = false;
			for (size_t k = 0; k < m_dimZ; k++) {
				for (size_t j = 0; j < m_dimY; j++) {
					for (size_t i = 0; i < m_dimX; i++) {
						bool valid = false;
						for (int ii = -radius; ii <= radius; ii++) {
							for (int jj = -radius; jj <= radius; jj++) {
								for (int kk = -radius; kk <= radius; kk++) {
									vec3i loc((int)i + ii, (int)j + jj, (int)k + kk);
									if (isValidCoordinate(loc) && (*this)(loc).weight >= weightThresh) {
										valid = true;
										break;
									}
								}
								if (valid) break;
							}
							if (valid) break;
						}
						if (valid) {
							if (checkDistToNeighborAndUpdate(i, j, k, bKeepSigns)) {
								hasUpdate = true;
							}
						}
						//if (checkDistToNeighborAndUpdate(i, j, k, bKeepSigns)) {
						//	hasUpdate = true;
						//}
					}
				}
			}

			if (!hasUpdate) break;
		}
	}

	//! bools checks if there is a neighbor with a smaller distance (+ the dist to the current voxel); if then it updates the distances and returns true
	bool checkDistToNeighborAndUpdate(size_t x, size_t y, size_t z, const bool bKeepSigns) {
		bool foundBetter = false;
		for (size_t k = 0; k < 3; k++) {
			for (size_t j = 0; j < 3; j++) {
				for (size_t i = 0; i < 3; i++) {
					if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
					vec3ul n(x - 1 + i, y - 1 + j, z - 1 + k);
					if (isValidCoordinate(n.x, n.y, n.z)) {
						float d = (vec3f((float)x, (float)y, (float)z) - vec3f((float)n.x, (float)n.y, (float)n.z)).length();
						float nSDF = (*this)(n.x, n.y, n.z).sdf;

						int sgn = bKeepSigns ? math::sign((*this)(x, y, z).sdf) : math::sign(nSDF);
						if (sgn != 0) {	//don't know that to do in this case...
							float dToN = bKeepSigns ? std::fabs(nSDF) + d : nSDF + sgn*d;

							if (std::abs(dToN) < std::abs((*this)(x, y, z).sdf)) {
								(*this)(x, y, z).sdf = bKeepSigns ? sgn*dToN : dToN;
								foundBetter = true;
							}
						}
					}
				}
			}
		}
		return foundBetter;

	}

	//!debug 
	void propagateColors(unsigned int numIter, const bool bKeepSigns, uchar colorThresh = 15, float sdfThresh = -std::numeric_limits<float>::infinity()) {
		const int radius = 2; // for weight filtering
		for (unsigned int iter = 0; iter < numIter; iter++) {
			bool hasUpdate = false;

			for (size_t k = 0; k < m_dimZ; k++) {
				for (size_t j = 0; j < m_dimY; j++) {
					for (size_t i = 0; i < m_dimX; i++) {
						const auto& v = (*this)(i, j, k);
						if (sdfThresh != -std::numeric_limits<float>::infinity() && std::fabs(v.sdf) > sdfThresh) continue;
						const auto& color = v.color;
						if (color.r > colorThresh || color.g > colorThresh || color.b > colorThresh)
							continue;

						if (checkDistToNeighborAndUpdateColor(i, j, k, bKeepSigns, colorThresh)) {
							hasUpdate = true;
						}
					}
				}
			}
			//std::cout << "waiting..." << std::endl;
			//getchar();

			if (!hasUpdate) break;
		}
	}

	//!debug 
	bool checkDistToNeighborAndUpdateColor(size_t x, size_t y, size_t z, const bool bKeepSigns, uchar colorThresh) {

		const bool bDebug = false;// (vec3f(x, y, z) - vec3f(100, 20, 50)).length() < 3;

		bool foundBetter = false;
		std::vector<std::pair<vec3ul, std::pair<float, vec3uc>>> candidates;
		float cur = (*this)(x, y, z).sdf;

		if (bDebug)std::cout << "checkDistToNeighborAndUpdateColor[" << x << ", " << y << ", " << z << "] (" << cur << " | " << vec3i((*this)(x, y, z).color) << ")" << std::endl;

		for (size_t k = 0; k < 3; k++) {
			for (size_t j = 0; j < 3; j++) {
				for (size_t i = 0; i < 3; i++) {
					if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
					vec3ul n(x - 1 + i, y - 1 + j, z - 1 + k);
					if (isValidCoordinate(n.x, n.y, n.z)) {
						const Voxel& vn = (*this)(n.x, n.y, n.z);

						if (bDebug) std::cout << "\t[" << x << ", " << y << ", " << z << "] (" << vn.sdf << " | " << vec3i(vn.color) << ")" << std::endl;
						if (vn.color.r > colorThresh || vn.color.g > colorThresh || vn.color.b > colorThresh)
							candidates.push_back(std::make_pair(n, std::make_pair(std::fabs(vn.sdf - cur), vn.color)));

						//float d = (vec3f((float)x, (float)y, (float)z) - vec3f((float)n.x, (float)n.y, (float)n.z)).length();
						//int sgn = bKeepSigns ? math::sign((*this)(x, y, z).sdf) : math::sign(vn.sdf);
						//if (sgn != 0) {	//don't know that to do in this case...
						//	float dToN = bKeepSigns ? std::fabs(vn.sdf) + d : vn.sdf + sgn*d;
						//	if (std::abs(dToN) < std::abs((*this)(x, y, z).sdf)) {
						//		(*this)(x, y, z).sdf = bKeepSigns ? sgn*dToN : dToN;
						//		foundBetter = true;
						//	}
						//}
					} // valid neighbor coord
				} //i
			} //j
		} //k
		if (!candidates.empty()) {
			std::sort(candidates.begin(), candidates.end(), [](const std::pair<vec3ul, std::pair<float, vec3uc>>& a, const std::pair<vec3ul, std::pair<float, vec3uc>>& b) {
				return a.second.first < b.second.first;
			});
			(*this)(x, y, z).color = candidates.front().second.second;
			foundBetter = true;
			if (bDebug)std::cout << "checkDistToNeighborAndUpdateColor[" << x << ", " << y << ", " << z << "] FoundBetter: (" << (*this)(x, y, z).sdf << " | " << vec3i((*this)(x, y, z).color) << ")" << std::endl;
		}
		return foundBetter;

	}

	vec3f getSurfaceNormal(size_t x, size_t y, size_t z) const {
		float SDFx = (*this)(x + 1, y, z).sdf - (*this)(x - 1, y, z).sdf;
		float SDFy = (*this)(x, y + 1, z).sdf - (*this)(x, y - 1, z).sdf;
		float SDFz = (*this)(x, y, z + 1).sdf - (*this)(x, y, z - 1).sdf;
		if (SDFx == 0 && SDFy == 0 && SDFz == 0) {// Don't divide by zero!
			return vec3f(SDFx, SDFy, SDFz);
		}
		else {
			return vec3f(SDFx, SDFy, SDFz).getNormalized();
		}
	}

	mat3f getNormalCovariance(int x, int y, int z, int radius, float weightThreshold, float sdfThreshold) const {
		// Compute neighboring surface normals
		std::vector<vec3f> normals;
		for (int k = -radius; k <= radius; k++)
			for (int j = -radius; j <= radius; j++)
				for (int i = -radius; i <= radius; i++)
					if ((*this)(x + i, y + j, z + k).weight >= weightThreshold && std::abs((*this)(x + i, y + j, z + k).sdf) < sdfThreshold)
						normals.push_back(getSurfaceNormal(x + i, y + j, z + k));

		// Find covariance matrix
		float Ixx = 0; float Ixy = 0; float Ixz = 0;
		float Iyy = 0; float Iyz = 0; float Izz = 0;
		for (int i = 0; i < normals.size(); i++) {
			Ixx = Ixx + normals[i].x*normals[i].x;
			Ixy = Ixy + normals[i].x*normals[i].y;
			Ixz = Ixz + normals[i].x*normals[i].z;
			Iyy = Iyy + normals[i].y*normals[i].y;
			Iyz = Iyz + normals[i].y*normals[i].z;
			Izz = Izz + normals[i].z*normals[i].z;
		}
		float scale = 10.0f / ((float)normals.size()); // Normalize and upscale
		return mat3f(Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz)*scale;
	}

	Voxel getVoxel(const vec3f& worldPos) const {
		vec3i voxelPos = worldToVoxel(worldPos);

		if (isValidCoordinate(voxelPos.x, voxelPos.y, voxelPos.z)) {
			return (*this)(voxelPos.x, voxelPos.y, voxelPos.z);
		}
		else {
			return Voxel();
		}
	}


	float getDepthMin() const {
		return m_depthMin;
	}

	float getDepthMax() const {
		return m_depthMax;
	}

	float getTruncation(float d) const {
		return m_trunaction + d * m_truncationScale;
	}

	float getMaxTruncation() const {
		return getTruncation(m_depthMax);
	}
private:

	float frac(float val) const {
		return (val - floorf(val));
	}

	vec3f frac(const vec3f& val) const {
		return vec3f(frac(val.x), frac(val.y), frac(val.z));
	}

	BoundingBox3<int> computeFrustumBounds(const mat4f& intrinsic, const mat4f& rigidTransform, unsigned int width, unsigned int height) const {

		std::vector<vec3f> cornerPoints(8);

		cornerPoints[0] = depthToSkeleton(intrinsic, 0, 0, m_depthMin);
		cornerPoints[1] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMin);
		cornerPoints[2] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMin);
		cornerPoints[3] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMin);

		cornerPoints[4] = depthToSkeleton(intrinsic, 0, 0, m_depthMax);
		cornerPoints[5] = depthToSkeleton(intrinsic, width - 1, 0, m_depthMax);
		cornerPoints[6] = depthToSkeleton(intrinsic, width - 1, height - 1, m_depthMax);
		cornerPoints[7] = depthToSkeleton(intrinsic, 0, height - 1, m_depthMax);

		BoundingBox3<int> box;
		for (unsigned int i = 0; i < 8; i++) {

			vec3f pl = math::floor(rigidTransform * cornerPoints[i]);
			vec3f pu = math::ceil(rigidTransform * cornerPoints[i]);
			box.include(worldToVoxel(pl));
			box.include(worldToVoxel(pu));
		}

		box.setMin(math::max(box.getMin(), 0));
		box.setMax(math::min(box.getMax(), vec3i((int)getDimX() - 1, (int)getDimY() - 1, (int)getDimZ() - 1)));

		return box;
	}

	static vec3f depthToSkeleton(const mat4f& intrinsic, unsigned int ux, unsigned int uy, float depth) {
		if (depth == 0.0f || depth == -std::numeric_limits<float>::infinity()) return vec3f(-std::numeric_limits<float>::infinity());

		float x = ((float)ux - intrinsic(0, 2)) / intrinsic(0, 0);
		float y = ((float)uy - intrinsic(1, 2)) / intrinsic(1, 1);

		return vec3f(depth*x, depth*y, depth);
	}

	static vec3f skeletonToDepth(const mat4f& intrinsics, const vec3f& p) {

		float x = (p.x * intrinsics(0, 0)) / p.z + intrinsics(0, 2);
		float y = (p.y * intrinsics(1, 1)) / p.z + intrinsics(1, 2);

		return vec3f(x, y, p.z);
	}

	float m_voxelSize;
	mat4f m_worldToGrid;
	mat4f m_gridToWorld; //inverse of worldToGrid
	float m_depthMin;
	float m_depthMax;
	OBB3f m_sceneBoundsGrid;


	float			m_trunaction;
	float			m_truncationScale;
	unsigned int	m_weightUpdate;
};