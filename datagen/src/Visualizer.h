#pragma once 

#include "ScansDirectory.h"

class Visualizer : public ApplicationCallback
{
public:
	void init(ApplicationData &app);
	void render(ApplicationData &app);
	void keyDown(ApplicationData &app, UINT key);
	void keyPressed(ApplicationData &app, UINT key);
	void mouseDown(ApplicationData &app, MouseButtonType button);
	void mouseMove(ApplicationData &app);
	void mouseWheel(ApplicationData &app, int wheelDelta);
	void resize(ApplicationData &app);

	void process(ApplicationData& app, float scaleBounds = 1.0f);
	void generateCompleteFrames(ApplicationData& app);

	static TriMeshf visualizeGrid(const Grid3f& grid, float invalidVal = -std::numeric_limits<float>::infinity(),
		const mat4f& voxelToWorld = mat4f::identity(), bool withNormals = false, bool bVerbose = false);

private:

	void generateCompleteFrames(const Scene& scene, std::vector<unsigned int>& completeFrames, bool bMatterportData) {
		completeFrames.clear();
		if (!bMatterportData) {
			completeFrames.resize(scene.getNumFrames());
			for (unsigned int i = 0; i < completeFrames.size(); i++) completeFrames[i] = i;
			return;
		}
		// matterport - filter out cameras not viewing the scene
		scene.computeTrajFramesInScene(completeFrames);
	}


	void generateIncompleteFramesMatterport(const Scene& scene, const std::vector<unsigned int>& completeFrames,
		float chanceDropFrame,  std::vector<unsigned int>& incompleteFrames,
		const std::unordered_set<unsigned int>& keepFramesSet = std::unordered_set<unsigned int>(),
		const std::unordered_set<unsigned int>& dontKeepFramesSet = std::unordered_set<unsigned int>()) {
		incompleteFrames.clear();
		//special case matterport frame characteristics different (randomly drop frames instead of dropping consecutive)
		for (unsigned int f : completeFrames) {
			if (keepFramesSet.find(f) != keepFramesSet.end())
				incompleteFrames.push_back(f);
			else if (dontKeepFramesSet.find(f) != dontKeepFramesSet.end())
				continue;
			else if (math::randomUniform(0.0f, 1.0f) > chanceDropFrame)
				incompleteFrames.push_back(f);
		}
	}

	ScansDirectory m_scans;

	Scene m_scene;
	bool m_bMatterport;

	D3D11Font m_font;
	FrameTimer m_timer;

	Cameraf m_camera;

	std::vector<std::vector<Cameraf>> m_recordedCameras;
	bool m_bEnableRecording;
	bool m_bEnableAutoRotate;
};