
#include "stdafx.h"

#include "VoxelGrid.h"


void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const ColorImageR8G8B8& color)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());
	const bool bHasColor = color.getNumPixels() > 0;

	//std::cout << std::endl << "voxelBounds: (" << voxelBounds.getMin() << ") (" << voxelBounds.getMax() << ")" << std::endl;

	//std::cout << "camera to world" << std::endl << cameraToWorld << std::endl;
	//std::cout << "world to camera" << std::endl << worldToCamera << std::endl;


	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {
				////debugging
				//const bool bDebug2 = i == 30 && j == 30 && k == 2;
				//if (bDebug2) std::cout << std::endl << "[" << i << "," << j << "," << k << "] | " << m_sceneBoundsGrid  << " => " << m_sceneBoundsGrid.intersects(vec3f((float)i, (float)j, (float)k)) << std::endl;
				////debugging
				if (!m_sceneBoundsGrid.intersects(vec3f((float)i, (float)j, (float)k))) continue; // not in obb of scene

				//transform to current frame
				vec3f pf = worldToCamera * voxelToWorld(vec3i(i, j, k));
				//vec3f p = worldToCamera * (m_gridToWorld * ((vec3f(i, j, k) + 0.5f)));

				//project into depth image
				vec3f p = skeletonToDepth(intrinsic, pf);

				vec3i pi = math::round(p);

				////debugging
				//const bool bDebug = i == 30 && j == 30 && (k == 4 || k == 3 || k == 2); //TODO WHY IS k=2 NEVER UPDATED
				////const bool bDebug2 = i == 30 && j == 30 && k == 2;
				//if (bDebug2) std::cout << "[" << i << "," << j << "," << k << "] => u,v (" << pi << ")" << std::endl;
				////debugging

				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					float d = depthImage(pi.x, pi.y);
					//if (bDebug2) std::cout << "[" << i << "," << j << "," << k << "] => depth (" << d << ")" << std::endl;

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);
						//if (bDebug2) std::cout << "[" << i << "," << j << "," << k << "] => sdf " << sdf << ", p.z " << p.z << ", d " << d << std::endl;

						//if (i == 110 && j == 36 && k == 4)
						//	std::cout << "at voxel (" << i << ", " << j << ", " << k << ") with depth " << d << " => sdf " << sdf << ", truncation " << truncation << std::endl;

						////if (std::abs(sdf) < truncation) {
						//if (sdf > -truncation) {
						//	Voxel& v = (*this)(i, j, k);
						//	if (sdf >= 0.0f || v.sdf <= 0.0f) {
						//		v.sdf = (v.sdf * (float)v.weight + sdf * (float)m_weightUpdate) / (float)(v.weight + m_weightUpdate);
						//		v.weight = (uchar)std::min((int)v.weight + (int)m_weightUpdate, (int)std::numeric_limits<unsigned char>::max());
						//	}
						//	//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						//}
						//if (std::abs(sdf) < truncation) {
						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);

							Voxel& v = (*this)(i, j, k);
							//if (bDebug) {
							//	std::cout << "updating (" << i << ", " << j << ", " << k << "); current (" << v.sdf << ", " << (int)v.weight << ")\t(sdf,d,p.z) = (" << sdf << ", " << d << ", " << p.z << ")" << std::endl;
							//}
							if (v.sdf == -std::numeric_limits<float>::infinity()) {
								v.sdf = sdf;
								if (bHasColor) v.color = color(pi.x, pi.y);
							}
							else {
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
								if (bHasColor) v.color = vec3uc(0.5f + 0.5f * vec3f(v.color) + 0.5f * vec3f(color(pi.x, pi.y)));
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());

							//if (i == 120 && j == 10 && k == 60) {//if (bDebug) {
							//	std::cout << "updated (" << i << ", " << j << ", " << k << ") to (" << v.sdf << ", " << (int)v.weight << ")" << std::endl;
							//	//std::cout << "\t==> (" << v.sdf << ", " << (int)v.weight << ")" << std::endl;
							//	FreeImageWrapper::saveImage("depth_" + std::to_string(v.sdf) + ".png", ColorImageR32G32B32(depthImage));
							//	int a = 5;
							//}
							//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						}
					}
				}

			}
		}
	}
}


void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& mask)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f p = worldToCamera * voxelToWorld(vec3i(i, j, k));
				//vec3f p = worldToCamera * (m_gridToWorld * ((vec3f(i, j, k) + 0.5f)));

				//project into depth image
				p = skeletonToDepth(intrinsic, p);

				vec3i pi = math::round(p);
				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					const float d = depthImage(pi.x, pi.y);
					unsigned char maskVal = mask(pi.x, pi.y);

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);

						////if (std::abs(sdf) < truncation) {
						//if (sdf > -truncation) {
						//	Voxel& v = (*this)(i, j, k);
						//	if (sdf >= 0.0f || v.sdf <= 0.0f) {
						//		v.sdf = (v.sdf * (float)v.weight + sdf * (float)m_weightUpdate) / (float)(v.weight + m_weightUpdate);
						//		v.weight = (uchar)std::min((int)v.weight + (int)m_weightUpdate, (int)std::numeric_limits<unsigned char>::max());
						//	}
						//	//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						//}
						//if (std::abs(sdf) < truncation) {
						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);

							Voxel& v = (*this)(i, j, k);
							if (v.sdf == -std::numeric_limits<float>::infinity()) {
								v.sdf = sdf;
							}
							else {
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());

							{
								if (maskVal == 2 && v.color.b == 0) v.color.b = maskVal;
								else if (maskVal == 1 && v.color.b == 2) v.color.b = maskVal;
							}
						} // sdf > -truncation
					} // valid depth value (in min/max range)
				} // in depth image bounds

			} //i
		} //j
	} //k
}

/*void VoxelGrid::integrate(const mat4f& intrinsic, const mat4f& cameraToWorld, const DepthImage32& depthImage, const BaseImage<unsigned char>& semantics)
{
	const mat4f worldToCamera = cameraToWorld.getInverse();
	BoundingBox3<int> voxelBounds = computeFrustumBounds(intrinsic, cameraToWorld, depthImage.getWidth(), depthImage.getHeight());

	for (int k = voxelBounds.getMinZ(); k <= voxelBounds.getMaxZ(); k++) {
		for (int j = voxelBounds.getMinY(); j <= voxelBounds.getMaxY(); j++) {
			for (int i = voxelBounds.getMinX(); i <= voxelBounds.getMaxX(); i++) {

				//transform to current frame
				vec3f p = worldToCamera * voxelToWorld(vec3i(i, j, k));
				//vec3f p = worldToCamera * (m_gridToWorld * ((vec3f(i, j, k) + 0.5f)));

				//project into depth image
				p = skeletonToDepth(intrinsic, p);

				vec3i pi = math::round(p);
				if (pi.x >= 0 && pi.y >= 0 && pi.x < (int)depthImage.getWidth() && pi.y < (int)depthImage.getHeight()) {
					const float d = depthImage(pi.x, pi.y);
					unsigned char sem = semantics(pi.x, pi.y);
					if (sem == 0) sem = 255;

					//check for a valid depth range
					if (d != depthImage.getInvalidValue() && d >= m_depthMin && d <= m_depthMax) {

						//update free space counter if voxel is in front of observation
						if (p.z < d) {
							(*this)(i, j, k).freeCtr++;
						}

						//compute signed distance; positive in front of the observation
						float sdf = d - p.z;
						float truncation = getTruncation(d);

						////if (std::abs(sdf) < truncation) {
						//if (sdf > -truncation) {
						//	Voxel& v = (*this)(i, j, k);
						//	if (sdf >= 0.0f || v.sdf <= 0.0f) {
						//		v.sdf = (v.sdf * (float)v.weight + sdf * (float)m_weightUpdate) / (float)(v.weight + m_weightUpdate);
						//		v.weight = (uchar)std::min((int)v.weight + (int)m_weightUpdate, (int)std::numeric_limits<unsigned char>::max());
						//	}
						//	//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						//}
						//if (std::abs(sdf) < truncation) {
						if (sdf > -truncation) {
							if (sdf >= 0.0f) {
								sdf = fminf(truncation, sdf);
							}
							else {
								sdf = fmaxf(-truncation, sdf);
							}
							const float integrationWeightSample = 3.0f;
							const float depthWorldMin = 0.4f;
							const float depthWorldMax = 4.0f;
							float depthZeroOne = (d - depthWorldMin) / (depthWorldMax - depthWorldMin);
							float weightUpdate = std::max(integrationWeightSample * 1.5f * (1.0f - depthZeroOne), 1.0f);

							Voxel& v = (*this)(i, j, k);
							if (v.sdf == -std::numeric_limits<float>::infinity()) {
								v.sdf = sdf;
							}
							else {
								v.sdf = (v.sdf * (float)v.weight + sdf * weightUpdate) / (float)(v.weight + weightUpdate);
							}
							v.weight = (uchar)std::min((int)v.weight + (int)weightUpdate, (int)std::numeric_limits<unsigned char>::max());

							//const vec3ui loc(42, 53, 129);
							//if (i == loc.x && j == loc.y && k == loc.z && (std::fabs(v.sdf) <= m_voxelSize) && sem == 7) {
							//	std::cout << loc << std::endl;
							//	std::cout << "image loc (" << pi.x << ", " << pi.y << ")" << std::endl;
							//	std::cout << "depth value " << d << " | p.z = " << p.z << std::endl;
							//	std::cout << "sdf = " << sdf << " -> v.sdf = " << v.sdf << std::endl;
							//	std::cout << "sem = " << (int)sem << std::endl;
							//	PointCloudf pcWorldSpace, pcVoxelSpace;
							//	for (const auto& p : depthImage) {
							//		if (p.value != -std::numeric_limits<float>::infinity()) {
							//			const vec3f campos = intrinsic.getInverse() * vec3f(p.x*p.value, p.y*p.value, p.value);
							//			const vec3f worldpos = cameraToWorld * campos;
							//			const vec3f voxelpos = m_gridToWorld.getInverse() * worldpos;
							//			pcWorldSpace.m_points.push_back(worldpos);
							//			pcVoxelSpace.m_points.push_back(voxelpos);
							//			const char ss = semantics(p.x, p.y);
							//			RGBColor c; 
							//			if (ss == 0) c = RGBColor(0, 0, 0);
							//			else if (ss == 255) c = RGBColor(128, 128, 128);
							//			else c = RGBColor::colorPalette((unsigned int)ss);
							//			const vec4f color(vec3f(c), 1.0f);
							//			pcWorldSpace.m_colors.push_back(color);
							//			pcVoxelSpace.m_colors.push_back(color);
							//		}
							//	}
							//	PointCloudIOf::saveToFile("pcWorld.ply", pcWorldSpace);
							//	PointCloudIOf::saveToFile("pcVoxel.ply", pcVoxelSpace);
							//	PointCloudf pcWorldPoint, pcVoxelPoint;
							//	pcWorldPoint.m_points.push_back(cameraToWorld * (intrinsic.getInverse() * vec3f(pi.x * d, pi.y * d, d)));
							//	pcWorldPoint.m_colors.push_back(vec4f(1.0f, 0.0f, 0.0f, 1.0f));
							//	pcVoxelPoint.m_points.push_back(m_gridToWorld.getInverse() * cameraToWorld * (intrinsic.getInverse() * vec3f(pi.x * d, pi.y * d, d)));
							//	pcVoxelPoint.m_colors.push_back(vec4f(1.0f, 0.0f, 0.0f, 1.0f));
							//	PointCloudIOf::saveToFile("pcWorldPoint.ply", pcWorldPoint);
							//	PointCloudIOf::saveToFile("pcVoxelPoint.ply", pcVoxelPoint);
							//	getchar();
							//}

							if (std::fabs(v.sdf) <= 1.1f * m_voxelSize) {
								if (std::fabs(sdf) <= 1.1 * m_voxelSize && (v.color.r == 0 || (sem != 0 && sem != 255))) {
									v.color.r = sem;
								}
								else {
									v.color.r = 255;
								}
							}
							//std::cout << "v: " << v.sdf << " " << (int)v.weight << std::endl;
						}
					}
				}

			}
		}
	}
}*/

TriMeshf VoxelGrid::computeSemanticsMesh(float sdfThresh) const {
	TriMeshf triMesh;

	// Pre-allocate space
	size_t nVoxels = 0;
	for (unsigned int z = 0; z < getDimZ(); z++) {
		for (unsigned int y = 0; y < getDimY(); y++) {
			for (unsigned int x = 0; x < getDimX(); x++) {
				if (std::fabs((*this)(x, y, z).sdf) < sdfThresh) nVoxels++;
			}
		}
	}
	size_t nVertices = nVoxels * 8; //no normals
	size_t nIndices = nVoxels * 12;
	triMesh.m_vertices.reserve(nVertices);
	triMesh.m_indices.reserve(nIndices);
	// Temporaries
	vec3f verts[24];
	vec3ui indices[12];
	vec3f normals[24];
	for (size_t z = 0; z < getDimZ(); z++) {
		for (size_t y = 0; y < getDimY(); y++) {
			for (size_t x = 0; x < getDimX(); x++) {
				const Voxel& v = (*this)(x, y, z);
				if (std::fabs(v.sdf) < sdfThresh) {
					vec3f p((float)x, (float)y, (float)z);
					vec3f pMin = p - 0.45f;//0.5f;
					vec3f pMax = p + 0.45f;//0.5f;
					bbox3f bb(pMin, pMax);
					bb.makeTriMesh(verts, indices);
					const unsigned char sem = v.color.r;

					unsigned int vertIdxBase = static_cast<unsigned int>(triMesh.m_vertices.size());
					for (size_t i = 0; i < 8; i++) {
						triMesh.m_vertices.emplace_back(verts[i]);
						if (sem == 0) {
							triMesh.m_vertices.back().color = vec4f(0.0f, 0.0f, 0.0f, 1.0f); //black for empty
						}
						else if (sem == 255) {
							triMesh.m_vertices.back().color = vec4f(0.5f, 0.5f, 0.5f, 1.0f); //gray for no annotation
						}
						else {
							RGBColor c = RGBColor::colorPalette((unsigned int)sem);
							triMesh.m_vertices.back().color = vec4f(vec3f(c.x, c.y, c.z) / 255.0f);
						}
					}
					for (size_t i = 0; i < 12; i++) {
						indices[i] += vertIdxBase;
						triMesh.m_indices.emplace_back(indices[i]);
					}
				}
			}
		}
	}
	triMesh.setHasColors(true);

	return triMesh;
}