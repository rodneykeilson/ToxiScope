import React, { useState } from 'react';
import { StyleSheet, Text, TextInput, View, ScrollView } from 'react-native';

import { PrimaryButton } from '../components/PrimaryButton';
import { useModel } from '../context/ModelContext';

// Label color mapping for visual feedback
const LABEL_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  toxic: { bg: '#fef2f2', text: '#dc2626', border: '#f87171' },
  severe_toxic: { bg: '#fdf2f8', text: '#be185d', border: '#f472b6' },
  obscene: { bg: '#fff7ed', text: '#ea580c', border: '#fb923c' },
  threat: { bg: '#fef3c7', text: '#d97706', border: '#fbbf24' },
  insult: { bg: '#f3e8ff', text: '#9333ea', border: '#c084fc' },
  identity_hate: { bg: '#ede9fe', text: '#7c3aed', border: '#a78bfa' },
  racism: { bg: '#e0e7ff', text: '#4f46e5', border: '#818cf8' },
};

interface PredictionResult {
  activeLabels: string[];
  rawScores: Record<string, number>;
}

export const HomeScreen = () => {
  const { status, ensureLoaded, predict, errorMessage } = useModel();
  const [sample, setSample] = useState('');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    if (!sample.trim()) {
      setError('Enter a comment to analyze.');
      setResult(null);
      return;
    }

    try {
      setIsSubmitting(true);
      setError('');
      await ensureLoaded();
      const prediction = await predict({ text: sample });
      setResult(prediction);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to analyze sample.';
      setError(message);
      setResult(null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderToxicityMeter = () => {
    if (!result) return null;
    
    // Calculate overall toxicity (max of all scores)
    const maxScore = Math.max(...Object.values(result.rawScores));
    const percentage = Math.round(maxScore * 100);
    
    let meterColor = '#22c55e'; // green
    if (percentage > 70) meterColor = '#ef4444'; // red
    else if (percentage > 40) meterColor = '#f59e0b'; // amber
    else if (percentage > 20) meterColor = '#eab308'; // yellow
    
    return (
      <View style={styles.meterContainer}>
        <Text style={styles.meterLabel}>Toxicity Level</Text>
        <View style={styles.meterTrack}>
          <View style={[styles.meterFill, { width: `${percentage}%`, backgroundColor: meterColor }]} />
        </View>
        <Text style={[styles.meterPercentage, { color: meterColor }]}>{percentage}%</Text>
      </View>
    );
  };

  const renderActiveLabels = () => {
    if (!result || result.activeLabels.length === 0) {
      return (
        <View style={styles.safeContainer}>
          <Text style={styles.safeText}>✓ No toxicity detected</Text>
        </View>
      );
    }
    
    return (
      <View style={styles.labelsContainer}>
        <Text style={styles.labelsTitle}>Detected Categories</Text>
        <View style={styles.labelsGrid}>
          {result.activeLabels.map((label) => {
            const colors = LABEL_COLORS[label] || { bg: '#f3f4f6', text: '#374151', border: '#9ca3af' };
            return (
              <View
                key={label}
                style={[styles.labelBadge, { backgroundColor: colors.bg, borderColor: colors.border }]}
              >
                <Text style={[styles.labelText, { color: colors.text }]}>
                  {label.replace(/_/g, ' ')}
                </Text>
                <Text style={[styles.labelScore, { color: colors.text }]}>
                  {(result.rawScores[label] * 100).toFixed(1)}%
                </Text>
              </View>
            );
          })}
        </View>
      </View>
    );
  };

  const renderScoreBreakdown = () => {
    if (!result) return null;
    
    const sortedScores = Object.entries(result.rawScores)
      .sort(([, a], [, b]) => b - a);
    
    return (
      <View style={styles.breakdownContainer}>
        <Text style={styles.breakdownTitle}>Score Breakdown</Text>
        {sortedScores.map(([label, score]) => {
          const percentage = Math.round(score * 100);
          const colors = LABEL_COLORS[label] || { bg: '#f3f4f6', text: '#94a3b8', border: '#475569' };
          const isActive = result.activeLabels.includes(label);
          
          return (
            <View key={label} style={styles.scoreRow}>
              <Text style={[styles.scoreLabelName, isActive && { color: colors.text, fontWeight: '600' }]}>
                {label.replace(/_/g, ' ')}
              </Text>
              <View style={styles.scoreBarTrack}>
                <View
                  style={[
                    styles.scoreBarFill,
                    { width: `${percentage}%`, backgroundColor: isActive ? colors.border : '#475569' },
                  ]}
                />
              </View>
              <Text style={[styles.scoreValue, isActive && { color: colors.text }]}>
                {percentage}%
              </Text>
            </View>
          );
        })}
      </View>
    );
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.header}>
        <Text style={styles.heading}>🔬 ToxiScope</Text>
        <Text style={styles.subtitle}>Gaming Community Toxicity Detection</Text>
      </View>
      
      <View style={styles.statusBar}>
        <View style={[styles.statusDot, { backgroundColor: status === 'ready' ? '#22c55e' : status === 'loading' ? '#f59e0b' : '#94a3b8' }]} />
        <Text style={styles.statusText}>
          {status === 'ready' ? 'Model Ready' : status === 'loading' ? 'Loading...' : 'Initializing'}
        </Text>
      </View>
      
      {errorMessage ? (
        <Text style={status === 'error' ? styles.errorText : styles.warningText}>
          {errorMessage}
        </Text>
      ) : null}

      <TextInput
        multiline
        value={sample}
        onChangeText={setSample}
        placeholder="Paste a gaming community comment to analyze..."
        style={styles.input}
        placeholderTextColor="#64748b"
      />

      <PrimaryButton
        label={isSubmitting ? 'Analyzing…' : '🔍 Analyze'}
        onPress={handleAnalyze}
        disabled={isSubmitting || status === 'loading'}
      />

      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      {result && (
        <View style={styles.resultsContainer}>
          {renderToxicityMeter()}
          {renderActiveLabels()}
          {renderScoreBreakdown()}
        </View>
      )}
      
      {!result && !error && (
        <View style={styles.placeholder}>
          <Text style={styles.placeholderText}>
            Enter a comment above and tap Analyze to detect toxicity.
          </Text>
          <Text style={styles.placeholderSubtext}>
            Detects: toxic, severe toxic, obscene, threat, insult, identity hate, racism
          </Text>
        </View>
      )}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 20,
  },
  heading: {
    fontSize: 28,
    fontWeight: '700',
    color: '#e2e8f0',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#64748b',
  },
  statusBar: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    padding: 8,
    backgroundColor: '#1e293b',
    borderRadius: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    color: '#94a3b8',
    fontSize: 13,
  },
  errorText: {
    color: '#f87171',
    marginBottom: 16,
    padding: 12,
    backgroundColor: 'rgba(248, 113, 113, 0.1)',
    borderRadius: 8,
  },
  warningText: {
    color: '#fbbf24',
    marginBottom: 16,
    padding: 12,
    backgroundColor: 'rgba(251, 191, 36, 0.1)',
    borderRadius: 8,
  },
  input: {
    minHeight: 120,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#334155',
    padding: 16,
    color: '#e2e8f0',
    backgroundColor: '#1e293b',
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 16,
    textAlignVertical: 'top',
  },
  resultsContainer: {
    marginTop: 24,
  },
  meterContainer: {
    padding: 16,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    marginBottom: 16,
  },
  meterLabel: {
    color: '#94a3b8',
    fontSize: 13,
    marginBottom: 8,
  },
  meterTrack: {
    height: 8,
    backgroundColor: '#334155',
    borderRadius: 4,
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    borderRadius: 4,
  },
  meterPercentage: {
    fontSize: 24,
    fontWeight: '700',
    marginTop: 8,
    textAlign: 'center',
  },
  safeContainer: {
    padding: 16,
    backgroundColor: 'rgba(34, 197, 94, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(34, 197, 94, 0.3)',
    marginBottom: 16,
  },
  safeText: {
    color: '#22c55e',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  labelsContainer: {
    padding: 16,
    backgroundColor: '#1e293b',
    borderRadius: 12,
    marginBottom: 16,
  },
  labelsTitle: {
    color: '#e2e8f0',
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 12,
  },
  labelsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  labelBadge: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  labelText: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'capitalize',
  },
  labelScore: {
    fontSize: 11,
    opacity: 0.8,
  },
  breakdownContainer: {
    padding: 16,
    backgroundColor: '#1e293b',
    borderRadius: 12,
  },
  breakdownTitle: {
    color: '#e2e8f0',
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 12,
  },
  scoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  scoreLabelName: {
    width: 100,
    fontSize: 12,
    color: '#94a3b8',
    textTransform: 'capitalize',
  },
  scoreBarTrack: {
    flex: 1,
    height: 6,
    backgroundColor: '#334155',
    borderRadius: 3,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  scoreBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  scoreValue: {
    width: 40,
    fontSize: 12,
    color: '#94a3b8',
    textAlign: 'right',
  },
  placeholder: {
    marginTop: 40,
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  placeholderText: {
    color: '#64748b',
    fontSize: 15,
    textAlign: 'center',
    marginBottom: 8,
  },
  placeholderSubtext: {
    color: '#475569',
    fontSize: 12,
    textAlign: 'center',
    lineHeight: 18,
  },
});
