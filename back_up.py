# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:34:32 2022

@author: Administrator
"""
def _plot_classification_history(self, output_width, output_height):
  fig = plt.figure(figsize=self._plot_figsize)

  for classification_history in self._pose_classification_filtered_history:
    y_cu = []
    y_ff = []
    for classification in classification_history:
      if classification is None:
        y_cu.append(None)
      elif 'CU' in classification:
        y_cu.append(classification['CU'])
      else:
        y_cu.append(0)

      if classification is None:
          y_ff.append(None)
      elif 'FF' in classification:
          y_ff.append(classification['FF'])
      else:
          y_ff.append(0)
    plt.plot(y_cu, linewidth=7,color='blue')
    plt.plot(y_ff, linewidth=7, color='red')

  plt.grid(axis='y', alpha=0.75)
  plt.xlabel('Frame')
  plt.ylabel('Confidence')
  plt.title('Classification history)
  plt.legend(loc='upper right')



def _plot_classification_history(self, output_width, output_height):
  fig = plt.figure(figsize=self._plot_figsize)

  for classification_history in [self._pose_classification_history,
                                 self._pose_classification_filtered_history]:
    y = []
    for classification in classification_history:
      if classification is None:
        y.append(None)
      elif self._class_name in classification:
        y.append(classification[self._class_name])
      else:
        y.append(0)
    plt.plot(y, linewidth=7)

  plt.grid(axis='y', alpha=0.75)
  plt.xlabel('Frame')
  plt.ylabel('Confidence')
  plt.title('Classification history for `{}`'.format(self._class_name))
  plt.legend(loc='upper right')